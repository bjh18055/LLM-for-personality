"""
Instagram 일기 데이터 → (per-account LMDB) → (merged LMDB cache) → PyTorch DataLoader.

3-stage 파이프라인:

    1. (계정별) ``build_lmdb_for_id()``
       raw JSON → ``data/instagram-{ID}-*/data.lmdb``
       각 계정의 authoritative source. 계정마다 한 번 빌드되면 끝.

    2. (조합별 merge cache) ``build_merged_lmdb()``
       per-account LMDB 들 → ``cache/diary-{ids_sorted_joined}.lmdb``
       fetch 할 때마다 ID 조합별로 단일 LMDB 파일이 만들어진다.
       per-account LMDB 의 mtime 이 merged 보다 새로우면 자동 재빌드.

    3. (학습) ``InstagramDiaryDataset(lmdb_path, ...)``
       merged LMDB 하나만 보고 동작. 라우팅 없음.

각 merged LMDB 의 ``__meta__`` 에 어떤 ID 조합 / 카운트 / source mtimes 가
박혀 있어 모델 체크포인트와 1:1 매칭 가능.

archived_posts.json 은 이 실험에서는 사용하지 않는다. posts_*.json 만 사용.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import lmdb
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from zoneinfo import ZoneInfo

    _TZ = ZoneInfo("Asia/Seoul")
except Exception:
    _TZ = None


DIARY_TAG = "[일기]"

DEFAULT_INSTAGRAM_IDS: List[str] = ["5082hjb"]
DEFAULT_CACHE_DIR = "cache"

LMDB_LEN_KEY = b"__len__"
LMDB_META_KEY = b"__meta__"

LMDB_DEFAULT_MAP_SIZE = 1 << 30

PER_ACCOUNT_LMDB_NAME = "data.lmdb"


WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


_ENV_CACHE: dict = {}


def _open_shared_readonly_env(lmdb_path: str):
    """프로세스·경로별로 하나의 readonly env 를 공유한다.

    LMDB 는 같은 프로세스가 같은 경로를 두 번 open 하는 것을 금지하므로
    (train+eval Dataset 동시 사용 시 충돌) ``(realpath, pid)`` 키로 캐싱한다.
    ``pid`` 를 키에 넣어 DataLoader 의 multi-worker fork 후에도 각 워커가
    자기 env 를 새로 연다.
    """
    key = (os.path.realpath(lmdb_path), os.getpid())
    env = _ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        _ENV_CACHE[key] = env
    return env


def _invalidate_env_cache(lmdb_path: str) -> None:
    """주어진 경로에 대한 모든 캐시된 env 를 닫고 캐시에서 제거.

    같은 프로세스에서 LMDB 를 rebuild 할 때 이전에 readonly 로 열려있던 env 가
    남아있으면 ``lmdb.open(write=...)`` 가 충돌한다. rebuild 전에 호출 필요.
    """
    real = os.path.realpath(lmdb_path)
    keys_to_remove = [k for k in list(_ENV_CACHE) if k[0] == real]
    for k in keys_to_remove:
        env = _ENV_CACHE.pop(k)
        try:
            env.close()
        except Exception:
            pass


@dataclass
class DiaryRecord:
    """단일 일기 항목 + 메타데이터. LMDB 에 저장되는 단위."""

    text: str
    timestamp: int
    source: str
    instagram_id: str = ""
    datetime_iso: str = ""
    date: str = ""
    year: int = 0
    month: int = 0
    day: int = 0
    weekday: str = ""
    weekday_idx: int = 0
    hour: int = 0
    minute: int = 0

    def to_training_text(self, tag: str = DIARY_TAG) -> str:
        return f"{tag}\n{self.text}"


def _to_dt(ts: int) -> datetime:
    if _TZ is not None:
        return datetime.fromtimestamp(ts, tz=_TZ)
    return datetime.fromtimestamp(ts)


def _enrich_metadata(rec: DiaryRecord) -> DiaryRecord:
    """``timestamp`` 로부터 날짜·요일·시각 메타데이터를 채워서 돌려준다."""
    dt = _to_dt(rec.timestamp)
    rec.datetime_iso = dt.isoformat()
    rec.date = dt.strftime("%Y-%m-%d")
    rec.year = dt.year
    rec.month = dt.month
    rec.day = dt.day
    rec.weekday_idx = dt.weekday()
    rec.weekday = WEEKDAY_NAMES[rec.weekday_idx]
    rec.hour = dt.hour
    rec.minute = dt.minute
    return rec


def _fix_mojibake(text: str) -> str:
    """Instagram 내보내기 JSON 의 latin-1 mojibake 를 UTF-8 한글로 복원."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def find_instagram_dir(data_dir: str, instagram_id: str) -> Optional[str]:
    """``data_dir`` 안에서 ``instagram-{id}-*`` 폴더 경로를 반환 (없으면 None)."""
    pattern = os.path.join(data_dir, f"instagram-{instagram_id}-*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def discover_instagram_ids(data_dir: str) -> List[str]:
    """``data_dir`` 안의 모든 ``instagram-{id}-*`` 폴더로부터 ID 목록 추출.

    폴더 이름 포맷은 ``instagram-{id}-{YYYY}-{MM}-{DD}-{hash}`` 를 가정한다.
    같은 ID 의 폴더가 여러 개 있으면 한 번만 포함되며, 결과는 사전순 정렬된다.
    """
    pattern = os.path.join(data_dir, "instagram-*")
    ids: List[str] = []
    seen = set()
    for path in sorted(glob.glob(pattern)):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        parts = name.split("-")
        if len(parts) < 3:
            continue
        ig_id = parts[1]
        if not ig_id or ig_id in seen:
            continue
        seen.add(ig_id)
        ids.append(ig_id)
    return sorted(ids)


def lmdb_path_for_id(data_dir: str, instagram_id: str) -> str:
    """``data/instagram-{id}-*/data.lmdb`` 경로를 반환 (폴더 없으면 FileNotFoundError)."""
    root = find_instagram_dir(data_dir, instagram_id)
    if root is None:
        raise FileNotFoundError(
            f"ID '{instagram_id}' 에 해당하는 폴더를 찾지 못했습니다: "
            f"{data_dir}/instagram-{instagram_id}-*"
        )
    return os.path.join(root, PER_ACCOUNT_LMDB_NAME)


def merged_lmdb_path(cache_dir: str, instagram_ids: Sequence[str]) -> str:
    """ID 조합에 대한 결정적인 merged LMDB 경로.

    ``cache_dir/diary-{id1+id2+...}.lmdb`` (ID 는 사전순 정렬).
    """
    if not instagram_ids:
        raise ValueError("instagram_ids 가 비어있습니다.")
    joined = "+".join(sorted(instagram_ids))
    return os.path.join(cache_dir, f"diary-{joined}.lmdb")


def _data_mdb_file(lmdb_path: str) -> str:
    return os.path.join(lmdb_path, "data.mdb")


def _media_dir(instagram_root: str) -> str:
    return os.path.join(instagram_root, "your_instagram_activity", "media")


def _iter_posts_records(
    payload: list, instagram_id: str, source: str, fix_mojibake: bool
) -> Iterable[DiaryRecord]:
    """posts_*.json 의 최상위 리스트를 순회하면서 일기 레코드를 yield."""
    for entry in payload:
        title = entry.get("title")
        timestamp = entry.get("creation_timestamp")

        if not title:
            media_list = entry.get("media") or []
            if media_list:
                title = media_list[0].get("title")
                if timestamp is None:
                    timestamp = media_list[0].get("creation_timestamp")

        if not title or timestamp is None:
            continue
        if fix_mojibake:
            title = _fix_mojibake(title)
        yield DiaryRecord(
            text=title.strip(),
            timestamp=int(timestamp),
            source=source,
            instagram_id=instagram_id,
        )


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_instagram_folder(
    data_dir: str,
    instagram_id: str,
    fix_mojibake: bool,
    verbose: bool,
) -> List[DiaryRecord]:
    """단일 ``instagram-{id}-*`` 폴더에서 일기 레코드를 추출 (계정 내 dedup + 시간순)."""
    ig_root = find_instagram_dir(data_dir, instagram_id)
    if ig_root is None:
        raise FileNotFoundError(
            f"ID '{instagram_id}' 에 해당하는 폴더를 찾지 못했습니다."
        )

    media_dir = _media_dir(ig_root)
    if not os.path.isdir(media_dir):
        raise FileNotFoundError(f"media 폴더가 없습니다: {media_dir}")

    records: List[DiaryRecord] = []
    posts_files = sorted(glob.glob(os.path.join(media_dir, "posts_*.json")))
    if not posts_files:
        if verbose:
            print(f"[{instagram_id}] posts_*.json 이 없습니다.")
        return records

    for posts_path in posts_files:
        payload = _load_json(posts_path)
        source = os.path.basename(posts_path)
        count_before = len(records)
        if isinstance(payload, list):
            records.extend(
                _iter_posts_records(
                    payload,
                    instagram_id=instagram_id,
                    source=source,
                    fix_mojibake=fix_mojibake,
                )
            )
        else:
            if verbose:
                print(
                    f"[{instagram_id}] {source} 의 최상위가 리스트가 아닙니다. 건너뜀."
                )
            continue
        if verbose:
            print(
                f"[{instagram_id}] {source} → {len(records) - count_before}개 추출"
            )

    seen = set()
    deduped: List[DiaryRecord] = []
    for rec in records:
        if rec.text in seen:
            continue
        seen.add(rec.text)
        deduped.append(_enrich_metadata(rec))

    deduped.sort(key=lambda r: r.timestamp)

    if verbose:
        print(
            f"[{instagram_id}] 총 {len(records)}개 → 중복 제거 후 {len(deduped)}개 "
            f"(중복 {len(records) - len(deduped)}개 제거)"
        )
        if deduped:
            print(
                f"[{instagram_id}] 기간: {deduped[0].datetime_iso}"
                f"  ~  {deduped[-1].datetime_iso}"
            )

    return deduped


def _idx_key(i: int) -> bytes:
    return f"{i:08d}".encode("ascii")


def _remove_lmdb_files(lmdb_path: str) -> None:
    for name in ("data.mdb", "lock.mdb"):
        p = os.path.join(lmdb_path, name)
        if os.path.isfile(p):
            os.remove(p)


def _write_records_to_lmdb(
    lmdb_path: str,
    records: Sequence[DiaryRecord],
    meta: dict,
    map_size: int,
) -> None:
    """레코드 시퀀스를 LMDB 에 zero-pad 정수 키로 일괄 저장. 메타도 함께 저장."""
    _invalidate_env_cache(lmdb_path)
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=True, lock=True)
    try:
        with env.begin(write=True) as txn:
            for i, rec in enumerate(records):
                txn.put(_idx_key(i), pickle.dumps(asdict(rec)))
            txn.put(LMDB_LEN_KEY, str(len(records)).encode())
            txn.put(LMDB_META_KEY, json.dumps(meta, ensure_ascii=False).encode())
    finally:
        env.close()


def build_lmdb_for_id(
    data_dir: str,
    instagram_id: str,
    lmdb_path: Optional[str] = None,
    fix_mojibake: bool = True,
    map_size: int = LMDB_DEFAULT_MAP_SIZE,
    overwrite: bool = False,
    verbose: bool = True,
) -> str:
    """단일 계정 폴더로부터 per-account LMDB 를 빌드. 경로 반환.

    Args:
        data_dir: 인스타그램 내보내기 폴더들이 모여있는 상위 디렉터리.
        instagram_id: 빌드할 계정 ID prefix (예: ``"5082hjb"``).
        lmdb_path: 출력 LMDB 경로. ``None`` 이면 계정 폴더 안 ``data.lmdb``.
        fix_mojibake: Instagram 한글 mojibake 복원 여부.
        map_size: LMDB 최대 크기 (bytes).
        overwrite: 기존 LMDB 가 있어도 덮어쓸지.
        verbose: 진행 상황 출력 여부.
    """
    if lmdb_path is None:
        lmdb_path = lmdb_path_for_id(data_dir, instagram_id)

    if os.path.isdir(lmdb_path) and not overwrite:
        if verbose:
            print(f"[build_lmdb] 이미 존재, 건너뜀: {lmdb_path}")
        return lmdb_path

    if os.path.isdir(lmdb_path):
        _remove_lmdb_files(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)

    if verbose:
        print(f"[build_lmdb] 파싱 시작: {instagram_id}")
    records = _parse_instagram_folder(
        data_dir=data_dir,
        instagram_id=instagram_id,
        fix_mojibake=fix_mojibake,
        verbose=verbose,
    )
    if not records:
        raise RuntimeError(
            f"[{instagram_id}] 일기 레코드를 하나도 추출하지 못했습니다."
        )

    meta = {
        "kind": "per_account",
        "instagram_id": instagram_id,
        "fix_mojibake": fix_mojibake,
        "count": len(records),
        "first_timestamp": records[0].timestamp,
        "last_timestamp": records[-1].timestamp,
        "first_iso": records[0].datetime_iso,
        "last_iso": records[-1].datetime_iso,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_records_to_lmdb(lmdb_path, records, meta, map_size)

    if verbose:
        print(f"[build_lmdb] 저장 완료: {lmdb_path}  (총 {len(records)}개)")
    return lmdb_path


def build_lmdb_for_ids(
    data_dir: str,
    instagram_ids: Sequence[str],
    fix_mojibake: bool = True,
    map_size: int = LMDB_DEFAULT_MAP_SIZE,
    overwrite: bool = False,
    verbose: bool = True,
) -> List[str]:
    """ID 리스트의 모든 계정에 대해 per-account LMDB 를 빌드 (idempotent)."""
    paths: List[str] = []
    for ig_id in instagram_ids:
        paths.append(
            build_lmdb_for_id(
                data_dir=data_dir,
                instagram_id=ig_id,
                fix_mojibake=fix_mojibake,
                map_size=map_size,
                overwrite=overwrite,
                verbose=verbose,
            )
        )
    return paths


def _is_merged_stale(
    source_lmdb_paths: Sequence[str], merged_path: str, verbose: bool
) -> bool:
    """merged LMDB 가 없거나, source 중 하나라도 더 새로우면 True."""
    merged_data = _data_mdb_file(merged_path)
    if not os.path.isfile(merged_data):
        return True
    merged_mtime = os.path.getmtime(merged_data)
    for src in source_lmdb_paths:
        src_data = _data_mdb_file(src)
        if not os.path.isfile(src_data):
            if verbose:
                print(f"[stale] source LMDB missing: {src}")
            return True
        if os.path.getmtime(src_data) > merged_mtime:
            if verbose:
                print(f"[stale] {src} 가 merged 보다 새로움")
            return True
    return False


def build_merged_lmdb(
    data_dir: str,
    instagram_ids: Sequence[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    fix_mojibake: bool = True,
    map_size: int = LMDB_DEFAULT_MAP_SIZE,
    overwrite: bool = False,
    rebuild_per_account: bool = False,
    verbose: bool = True,
) -> str:
    """ID 조합에 해당하는 merged LMDB 를 만들고 경로 반환.

    내부적으로:
        1. 각 계정의 per-account LMDB 를 빌드 (없거나 ``rebuild_per_account``)
        2. ``cache_dir/diary-{ids_sorted_joined}.lmdb`` 경로 계산
        3. 모든 source mtime 을 merged mtime 과 비교 → stale 이면 재빌드,
           아니면 cache hit (``overwrite=True`` 면 무조건 재빌드)
        4. merged 안에는 timestamp 기준 글로벌 정렬된 레코드들이 zero-pad 정수 키로 저장됨.
        5. ``__meta__`` 에 instagram_ids, per-account 카운트, source mtimes, built_at 등 자기 기술.

    Returns:
        merged LMDB 경로.
    """
    if not instagram_ids:
        raise ValueError("instagram_ids 가 비어있습니다.")

    src_paths = build_lmdb_for_ids(
        data_dir=data_dir,
        instagram_ids=instagram_ids,
        fix_mojibake=fix_mojibake,
        map_size=map_size,
        overwrite=rebuild_per_account,
        verbose=verbose,
    )

    merged_path = merged_lmdb_path(cache_dir, instagram_ids)

    if not overwrite and not _is_merged_stale(src_paths, merged_path, verbose):
        if verbose:
            print(f"[build_merged_lmdb] cache hit: {merged_path}")
        return merged_path

    if os.path.isdir(merged_path):
        _remove_lmdb_files(merged_path)
    os.makedirs(merged_path, exist_ok=True)

    all_records: List[DiaryRecord] = []
    per_account_counts: dict = {}
    src_mtimes: dict = {}
    for ig_id, src in zip(instagram_ids, src_paths):
        recs = read_lmdb_records(src)
        per_account_counts[ig_id] = len(recs)
        src_mtimes[ig_id] = os.path.getmtime(_data_mdb_file(src))
        all_records.extend(recs)

    all_records.sort(key=lambda r: (r.timestamp, r.instagram_id))

    meta = {
        "kind": "merged",
        "instagram_ids": sorted(instagram_ids),
        "instagram_ids_input_order": list(instagram_ids),
        "per_account_counts": per_account_counts,
        "source_mtimes": src_mtimes,
        "fix_mojibake": fix_mojibake,
        "count": len(all_records),
        "first_iso": all_records[0].datetime_iso if all_records else "",
        "last_iso": all_records[-1].datetime_iso if all_records else "",
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_records_to_lmdb(merged_path, all_records, meta, map_size)

    if verbose:
        breakdown = ", ".join(f"{k}={v}" for k, v in per_account_counts.items())
        print(f"[build_merged_lmdb] 저장 완료: {merged_path}")
        print(f"  총 {len(all_records)}개 ({breakdown})")

    return merged_path


def read_lmdb_meta(lmdb_path: str) -> dict:
    """LMDB 의 ``__meta__`` 키를 읽어 dict 로 돌려준다."""
    env = _open_shared_readonly_env(lmdb_path)
    with env.begin() as txn:
        raw = txn.get(LMDB_META_KEY)
        if raw is None:
            return {}
        return json.loads(raw.decode())


def read_lmdb_records(lmdb_path: str) -> List[DiaryRecord]:
    """LMDB 의 모든 레코드를 ``DiaryRecord`` 리스트로 로드 (검사·디버그용)."""
    env = _open_shared_readonly_env(lmdb_path)
    with env.begin() as txn:
        raw = txn.get(LMDB_LEN_KEY)
        if raw is None:
            raise RuntimeError(f"LMDB 가 손상됨: {lmdb_path}")
        n = int(raw.decode())
        out: List[DiaryRecord] = []
        for i in range(n):
            buf = txn.get(_idx_key(i))
            if buf is None:
                raise RuntimeError(f"LMDB 가 손상됨: index {i} 없음 ({lmdb_path})")
            out.append(DiaryRecord(**pickle.loads(buf)))
        return out


class InstagramDiaryDataset(Dataset):
    """단일 LMDB (보통 merged) 를 보고 동작하는 Causal LM 학습용 Dataset.

    각 샘플은 ``{"input_ids", "attention_mask", "labels"}`` 를 반환하며,
    ``labels`` 는 ``input_ids`` 의 복사본이다. padding 위치는 기본적으로
    ``-100`` 으로 마스킹 (``ignore_pad_in_loss``).

    LMDB env 는 ``_open_shared_readonly_env`` 로 (realpath, pid) 별로 공유된다.
    """

    def __init__(
        self,
        lmdb_path: str,
        tokenizer,
        indices: Optional[Sequence[int]] = None,
        max_length: int = 512,
        diary_tag: str = DIARY_TAG,
        ignore_pad_in_loss: bool = True,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.diary_tag = diary_tag
        self.ignore_pad_in_loss = ignore_pad_in_loss

        env = _open_shared_readonly_env(self.lmdb_path)
        with env.begin() as txn:
            raw = txn.get(LMDB_LEN_KEY)
            if raw is None:
                raise RuntimeError(
                    f"LMDB 가 손상됨 (key={LMDB_LEN_KEY!r} 없음): {lmdb_path}"
                )
            total = int(raw.decode())

        if indices is None:
            self.indices: List[int] = list(range(total))
        else:
            self.indices = list(indices)
            for g in self.indices:
                if not 0 <= g < total:
                    raise IndexError(
                        f"index {g} 가 범위 [0, {total}) 밖입니다."
                    )

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def __len__(self) -> int:
        return len(self.indices)

    def get_record(self, i: int) -> DiaryRecord:
        """``i`` 번째 샘플의 raw ``DiaryRecord`` 를 반환 (디버그·메타 조회용)."""
        global_idx = self.indices[i]
        env = _open_shared_readonly_env(self.lmdb_path)
        with env.begin() as txn:
            buf = txn.get(_idx_key(global_idx))
        if buf is None:
            raise IndexError(
                f"LMDB 에 index {global_idx} 가 없습니다: {self.lmdb_path}"
            )
        return DiaryRecord(**pickle.loads(buf))

    def __getitem__(self, i: int) -> dict:
        record = self.get_record(i)
        text = record.to_training_text(self.diary_tag)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        if self.ignore_pad_in_loss:
            labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _split_indices(
    n: int, train_ratio: float, seed: int
) -> tuple[List[int], List[int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 는 (0, 1) 범위여야 합니다. got {train_ratio}")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(round(n * train_ratio))
    return sorted(indices[:n_train]), sorted(indices[n_train:])


def build_dataloaders(
    data_dir: str,
    tokenizer,
    instagram_ids: Optional[Sequence[str]] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_length: int = 512,
    batch_size: int = 4,
    eval_batch_size: Optional[int] = None,
    train_ratio: float = 0.9,
    num_workers: int = 0,
    seed: int = 42,
    diary_tag: str = DIARY_TAG,
    ignore_pad_in_loss: bool = True,
    *,
    fix_mojibake: bool = True,
    rebuild_lmdb: bool = False,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, InstagramDiaryDataset, InstagramDiaryDataset]:
    """``data_dir`` + ``instagram_ids`` 로부터 train / eval DataLoader 쌍을 생성.

    내부 흐름:
        per-account LMDB 빌드 (없거나 ``rebuild_lmdb``)
        → merged LMDB 빌드 (stale 이거나 ``rebuild_lmdb``)
        → Dataset 은 merged LMDB 단일 경로만 본다.

    Args:
        data_dir: 인스타그램 내보내기 폴더들이 모여있는 상위 디렉터리.
        tokenizer: HuggingFace AutoTokenizer 인스턴스.
        instagram_ids: 사용할 인스타그램 ID prefix 리스트. ``None`` 이면
            ``DEFAULT_INSTAGRAM_IDS``.
        cache_dir: merged LMDB 들이 저장될 디렉터리. 기본 ``"cache"``.
        max_length: 토큰 시퀀스 최대 길이.
        batch_size / eval_batch_size: 배치 크기.
        train_ratio: train split 비율.
        num_workers: DataLoader worker 수.
        seed: split 셔플용 random seed.
        diary_tag: 각 일기 텍스트 앞에 붙일 태그.
        ignore_pad_in_loss: padding 위치 label 을 -100 으로 마스킹할지.
        fix_mojibake: per-account LMDB 빌드 시 mojibake 복원 여부.
        rebuild_lmdb: per-account 와 merged 둘 다 강제 재빌드.
        verbose: 진행 상황 출력 여부.

    Returns:
        ``(train_loader, eval_loader, train_dataset, eval_dataset)`` 및
        Dataset 들은 같은 merged LMDB 를 공유 (다른 ``indices`` 로 분할).
    """
    if instagram_ids is None:
        instagram_ids = DEFAULT_INSTAGRAM_IDS

    merged_path = build_merged_lmdb(
        data_dir=data_dir,
        instagram_ids=instagram_ids,
        cache_dir=cache_dir,
        fix_mojibake=fix_mojibake,
        overwrite=rebuild_lmdb,
        rebuild_per_account=rebuild_lmdb,
        verbose=verbose,
    )

    env = _open_shared_readonly_env(merged_path)
    with env.begin() as txn:
        total = int(txn.get(LMDB_LEN_KEY).decode())

    train_idx, eval_idx = _split_indices(total, train_ratio, seed)
    if verbose:
        print(f"train: {len(train_idx)}개 / eval: {len(eval_idx)}개")

    train_dataset = InstagramDiaryDataset(
        lmdb_path=merged_path,
        tokenizer=tokenizer,
        indices=train_idx,
        max_length=max_length,
        diary_tag=diary_tag,
        ignore_pad_in_loss=ignore_pad_in_loss,
    )
    eval_dataset = InstagramDiaryDataset(
        lmdb_path=merged_path,
        tokenizer=tokenizer,
        indices=eval_idx,
        max_length=max_length,
        diary_tag=diary_tag,
        ignore_pad_in_loss=ignore_pad_in_loss,
    )

    if eval_batch_size is None:
        eval_batch_size = batch_size

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, eval_loader, train_dataset, eval_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer

    DATA_DIR = "data"
    INSTAGRAM_IDS = ["5082hjb"]
    CACHE_DIR = "cache"
    MODEL_NAME = "skt/kogpt2-base-v2"

    print(f"[1/4] LMDB 준비 (per-account → merged): {INSTAGRAM_IDS}")
    merged = build_merged_lmdb(
        data_dir=DATA_DIR,
        instagram_ids=INSTAGRAM_IDS,
        cache_dir=CACHE_DIR,
    )
    print(f"\n  merged path: {merged}")
    print(
        "  meta:",
        json.dumps(read_lmdb_meta(merged), ensure_ascii=False, indent=2),
    )

    print(f"\n[2/4] tokenizer 로드: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"  └ '{MODEL_NAME}' 로드 실패 ({e!r}). gpt2 로 폴백.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"\n[3/4] DataLoader 구성")
    train_loader, eval_loader, train_ds, eval_ds = build_dataloaders(
        data_dir=DATA_DIR,
        instagram_ids=INSTAGRAM_IDS,
        cache_dir=CACHE_DIR,
        tokenizer=tokenizer,
        max_length=512,
        batch_size=2,
        train_ratio=0.9,
        seed=42,
    )

    print(f"\n[4/4] 샘플 확인")
    sample = train_ds[0]
    rec = train_ds.get_record(0)
    print(
        f"  [메타] id={rec.instagram_id}  {rec.date} {rec.weekday} "
        f"{rec.hour:02d}:{rec.minute:02d}  (src={rec.source})"
    )
    print("  input_ids shape    :", tuple(sample["input_ids"].shape))
    print("  attention_mask sum :", int(sample["attention_mask"].sum()))
    preview = tokenizer.decode(sample["input_ids"][:80], skip_special_tokens=False)
    print("  decode(first 80 tok):")
    print("    " + preview.replace("\n", "\\n"))

    print(f"\n  train_loader 배치 개수: {len(train_loader)}")
    print(f"  eval_loader  배치 개수: {len(eval_loader)}")

    batch = next(iter(train_loader))
    print("\n  배치 shape:")
    for k, v in batch.items():
        print(f"    {k}: {tuple(v.shape)}  dtype={v.dtype}")

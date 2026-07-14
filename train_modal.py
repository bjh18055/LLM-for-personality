"""
Modal 서버리스 래퍼: 기존 train.py / dataset.py 를 그대로 GPU 잡으로 실행한다.

특징
    - 서버리스 A100 40GB. 학습이 도는 동안만 초 단위 과금, 끝나면 자동 종료.
    - 개인 Modal Volume 3개를 사용해 재업로드/재다운로드를 없앤다:
        * diary-data      : 일기 LMDB (한 번만 업로드)
        * diary-hf-cache  : Qwen2.5-7B 가중치 캐시 (첫 실행 후 재사용)
        * diary-output    : 학습 결과(LoRA adapter) 저장 → 로컬로 내려받음

준비 (최초 1회)
    pip install modal
    modal token new
    bash scripts/pack_data.sh                                   # diary-lmdb.tar.gz 생성
    modal volume create diary-data
    modal volume put diary-data diary-lmdb.tar.gz /diary-lmdb.tar.gz
    modal run train_modal.py::prepare                           # 볼륨 안에서 압축 해제

학습 (매번 이 한 줄)
    modal run train_modal.py --epochs 3
    # 하이퍼파라미터: --epochs --batch-size --lr --instagram-ids --lora-r --lora-alpha

결과 회수 (로컬 대상은 반드시 폴더로 — 끝에 / 또는 미리 mkdir)
    modal volume ls diary-output
    mkdir -p results && modal volume get diary-output <run_dir> results/
"""

import glob
import os
import shutil
import subprocess
import time

import modal

APP_NAME = "diary-lora"

# 학습 컨테이너 이미지: 프로젝트 파이썬 파일 + 학습 의존성.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.12.0",
        "transformers>=5.9.0",
        "accelerate>=1.0",
        "peft>=0.13",
        "lmdb>=2.2.0",
        "huggingface_hub",
    )
    .add_local_file("train.py", "/root/proj/train.py")
    .add_local_file("dataset.py", "/root/proj/dataset.py")
)

app = modal.App(APP_NAME, image=image)

# 재사용되는 개인 볼륨 (내 Modal 계정 전용, 공용 조직 아님).
data_vol = modal.Volume.from_name("diary-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("diary-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("diary-output", create_if_missing=True)

DATA_MOUNT = "/data"
HF_MOUNT = "/hf"
OUT_MOUNT = "/out"
WORKDIR = "/root/proj"


@app.function(volumes={DATA_MOUNT: data_vol}, timeout=600)
def prepare() -> None:
    """diary-data 볼륨에 올려둔 diary-lmdb.tar.gz 를 볼륨 안에서 압축 해제한다.

    결과 배치: /data/data/instagram-*/data.lmdb/... , /data/cache/*.lmdb/...
    최초 1회만 실행하면 되고, 데이터가 바뀔 때만 다시 실행한다.
    """
    tarball = os.path.join(DATA_MOUNT, "diary-lmdb.tar.gz")
    if not os.path.isfile(tarball):
        raise FileNotFoundError(
            f"{tarball} 가 없습니다. 먼저 실행:\n"
            f"  modal volume put diary-data diary-lmdb.tar.gz /diary-lmdb.tar.gz"
        )
    print(f"[prepare] 압축 해제: {tarball} → {DATA_MOUNT}")
    subprocess.run(["tar", "-xzf", tarball, "-C", DATA_MOUNT], check=True)
    data_vol.commit()
    print("[prepare] 완료. 내용:")
    for root, _dirs, files in os.walk(DATA_MOUNT):
        for f in files:
            if f == "data.mdb":
                print("   ", os.path.join(root, f))


def _stage_data_into_workdir() -> None:
    """볼륨의 LMDB 를 작업 디렉터리로 복사하고 mtime 을 정리한다.

    dataset.py 는 cwd 기준 상대경로 data/ , cache/ 를 본다. 또한 merged 캐시가
    per-account LMDB 보다 최신이어야 원본 JSON 재파싱 없이 cache hit 한다.
    """
    src_data = os.path.join(DATA_MOUNT, "data")
    src_cache = os.path.join(DATA_MOUNT, "cache")
    if not os.path.isdir(src_data) or not os.path.isdir(src_cache):
        raise FileNotFoundError(
            "볼륨에 data/ 또는 cache/ 가 없습니다. 먼저 "
            "`modal run train_modal.py::prepare` 를 실행하세요."
        )
    dst_data = os.path.join(WORKDIR, "data")
    dst_cache = os.path.join(WORKDIR, "cache")
    shutil.rmtree(dst_data, ignore_errors=True)
    shutil.rmtree(dst_cache, ignore_errors=True)
    shutil.copytree(src_data, dst_data)
    shutil.copytree(src_cache, dst_cache)

    # per-account(source) 를 먼저, merged 캐시를 나중에 touch → 캐시가 항상 최신.
    now = time.time()
    for root, _d, files in os.walk(dst_data):
        for f in files:
            if f == "data.mdb":
                os.utime(os.path.join(root, f), (now, now))
    later = now + 5
    for root, _d, files in os.walk(dst_cache):
        for f in files:
            if f == "data.mdb":
                os.utime(os.path.join(root, f), (later, later))


@app.function(
    gpu="A100",
    volumes={DATA_MOUNT: data_vol, HF_MOUNT: hf_vol, OUT_MOUNT: out_vol},
    timeout=2 * 60 * 60,
)
def train(
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-4,
    grad_accum_steps: int = 4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    instagram_ids: str = "all",
) -> str:
    """A100 위에서 train.py 를 accelerate 로 실행하고 결과를 diary-output 에 저장."""
    os.environ["HF_HOME"] = HF_MOUNT  # 모델 가중치를 볼륨에 캐시

    _stage_data_into_workdir()

    run_name = (
        f"lr{lr}_ep{epochs}_bs{batch_size}_ga{grad_accum_steps}"
        f"_r{lora_r}_a{lora_alpha}".replace(".", "p")
    )
    output_dir = os.path.join(OUT_MOUNT, run_name)

    cmd = [
        "accelerate", "launch", "--num_processes", "1", "train.py",
        "--data_dir", "./data",
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--grad_accum_steps", str(grad_accum_steps),
        "--max_length", str(max_length),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--instagram_ids", *instagram_ids.split(),
        "--logging_steps", "1",
        # LMDB env 는 fork 된 DataLoader 워커와 충돌하므로 워커를 끈다.
        # (데이터셋이 작아 num_workers>0 의 이득도 없음.)
        "--num_workers", "0",
    ]
    print("[train] 실행:", " ".join(cmd))
    subprocess.run(cmd, cwd=WORKDIR, check=True)

    out_vol.commit()

    # train.py 는 output_dir 뒤에 _YYYYMMDD-HHMMSS 를 붙이므로 실제 폴더명을 찾는다.
    matches = sorted(glob.glob(output_dir + "_*"))
    actual = os.path.basename(matches[-1]) if matches else run_name
    print(f"[train] 완료. 결과 저장(볼륨): /out/{actual}")
    print(
        f"[train] 로컬 다운로드: "
        f"mkdir -p results && modal volume get diary-output {actual} results/"
    )
    return actual


@app.local_entrypoint()
def main(
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-4,
    grad_accum_steps: int = 4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    instagram_ids: str = "all",
):
    run_name = train.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_accum_steps=grad_accum_steps,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        instagram_ids=instagram_ids,
    )
    print(f"\n실행 완료: {run_name}")
    print(
        f"결과 다운로드: mkdir -p results && "
        f"modal volume get diary-output {run_name} results/"
    )

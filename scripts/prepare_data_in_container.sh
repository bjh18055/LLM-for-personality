#!/usr/bin/env bash
set -euo pipefail

# VESSL 워크스페이스(컨테이너) 안에서 실행한다.
# 업로드된 diary-lmdb.tar.gz 를 프로젝트 루트에 풀어 data/·cache/ LMDB 를
# 제자리에 놓고, mtime 을 정리해 dataset.py 가 원본 JSON 재파싱 없이
# 곧바로 "cache hit" 하도록 만든다.
#
# 사용법 (컨테이너 안, 프로젝트 루트에서):
#   TARBALL=/root/diary-lmdb.tar.gz bash scripts/prepare_data_in_container.sh

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# 업로드 위치가 환경마다 다를 수 있어(홈, Jupyter cwd, 프로젝트 루트) 후보를 훑는다.
TARBALL="${TARBALL:-}"
if [ -z "$TARBALL" ]; then
  for cand in \
    "$HOME/diary-lmdb.tar.gz" \
    "/root/diary-lmdb.tar.gz" \
    "$PROJECT_ROOT/diary-lmdb.tar.gz" \
    "$(pwd)/diary-lmdb.tar.gz"; do
    if [ -f "$cand" ]; then TARBALL="$cand"; break; fi
  done
fi

if [ -z "$TARBALL" ] || [ ! -f "$TARBALL" ]; then
  echo "[ERROR] 업로드된 번들(diary-lmdb.tar.gz)을 찾지 못했습니다."
  echo "        Jupyter/VSCode 로 홈(~/)에 업로드했는지 확인하거나,"
  echo "        TARBALL=<경로> 로 직접 지정하세요."
  exit 1
fi

echo "[INFO] 번들 해제: $TARBALL → $PROJECT_ROOT"
tar -xzf "$TARBALL" -C "$PROJECT_ROOT"

# mtime 정리: per-account LMDB(source) 를 먼저 touch 하고, 그 다음 merged 캐시를
# touch 해서 merged 가 항상 더 최신이 되게 한다. (dataset.py 의 _is_merged_stale 는
# source mtime > merged mtime 이면 재빌드하려 하는데, 원본 JSON 이 없으면 실패하므로
# 반드시 merged 가 최신이어야 cache hit 이 보장된다.)
find "$PROJECT_ROOT/data" -type f -path '*/data.lmdb/data.mdb' -exec touch {} + 2>/dev/null || true
sleep 1
find "$PROJECT_ROOT/cache" -type f -path '*.lmdb/data.mdb' -exec touch {} + 2>/dev/null || true

echo "[INFO] 배치 완료. 확인:"
find "$PROJECT_ROOT/data" -path '*/data.lmdb/data.mdb' 2>/dev/null
find "$PROJECT_ROOT/cache" -path '*.lmdb/data.mdb' 2>/dev/null
echo "[INFO] 이제 학습을 실행할 수 있습니다: bash scripts/run_train.sh"

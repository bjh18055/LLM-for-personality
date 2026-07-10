#!/usr/bin/env bash
set -uo pipefail

# 워크스페이스(컨테이너) 안에서 백그라운드로 돌면서, 업로드된 데이터 번들이
# 나타나면 데이터를 배치하고 학습을 자동 실행한다.
# init-script 가 `nohup bash scripts/watch_and_train.sh > ~/train.out 2>&1 &` 로 띄운다.
#
# 감지 대상: $TARBALL (기본 ~/diary-lmdb.tar.gz). 웹 UI(Jupyter/VSCode)로 홈에
# 업로드하면 몇 초 안에 감지된다.

cd "$(dirname "$0")/.."

TARBALL="${TARBALL:-$HOME/diary-lmdb.tar.gz}"
WAIT_ITERS="${WAIT_ITERS:-4320}"   # 4320 * 5s = 6h 대기창

echo "[watch] $(date) 데이터 업로드 대기: $TARBALL"
for i in $(seq 1 "$WAIT_ITERS"); do
  [ -f "$TARBALL" ] && break
  sleep 5
done

if [ ! -f "$TARBALL" ]; then
  echo "[watch] 시간 초과: $TARBALL 가 업로드되지 않았습니다."
  echo "[watch] 업로드 후 수동 실행: TARBALL=$TARBALL bash scripts/prepare_data_in_container.sh && bash scripts/run_train.sh"
  exit 1
fi

echo "[watch] $(date) 데이터 감지됨. 배치 시작."
TARBALL="$TARBALL" bash scripts/prepare_data_in_container.sh

echo "===TRAIN_START=== $(date)"
bash scripts/run_train.sh
echo "===TRAIN_DONE=== $(date)"

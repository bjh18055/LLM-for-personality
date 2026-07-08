#!/usr/bin/env bash
set -euo pipefail

# 옵션2(런타임 데이터 전송) 방식으로 VESSL 인터랙티브 워크스페이스를 띄운다.
#   - 코드: 공개 GitHub repo 에서 git clone (init-script)
#   - 데이터: 최소 LMDB 번들(diary-lmdb.tar.gz)을 워크스페이스에만 업로드
#            (영구 공용 Dataset 에 남기지 않음 → 프라이버시 보호)
#   - 셋업: init-script 가 clone + pip install + 데이터 배치까지 자동 수행
#
# 사용 예:
#   REPO_URL=https://github.com/<you>/<repo>.git bash scripts/run_workspace.sh
#
# 이후 접속:
#   vessl workspace ssh          # 또는 웹 UI 의 Jupyter/VSCode
#   cd ~/LLM-for-personality && bash scripts/run_train.sh

cd "$(dirname "$0")/.."

REPO_URL="${REPO_URL:-https://github.com/bjh18055/LLM-for-personality.git}"
REPO_DIR="${REPO_DIR:-LLM-for-personality}"
BRANCH="${BRANCH:-main}"

CLUSTER="${CLUSTER:-snu-eng-dgx}"
RESOURCE="${RESOURCE:-a100-1}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.05-py3}"
WS_NAME="${WS_NAME:-diary-lora-$(date +%m%d-%H%M)}"
MAX_HOURS="${MAX_HOURS:-6}"

TARBALL="diary-lmdb.tar.gz"
REMOTE_TARBALL="/root/${TARBALL}"

# --- 사전 점검 ---
if ! command -v vessl >/dev/null 2>&1; then
  echo "[ERROR] 'vessl' 명령을 찾을 수 없습니다. VESSL 환경을 활성화하세요."
  exit 1
fi
if ! vessl whoami >/dev/null 2>&1; then
  echo "[ERROR] VESSL 인증이 안 되어 있습니다. 'vessl configure' 실행 후 재시도."
  exit 1
fi

# --- 데이터 번들 생성 (최소 LMDB) ---
echo "[INFO] 데이터 번들 생성..."
bash scripts/pack_data.sh

# --- init-script: 컨테이너 시작 시 자동 셋업 ---
# 업로드된 번들이 늦게 도착할 수 있어 잠깐 대기 후 데이터 배치.
INIT_SCRIPT=$(cat <<INIT
set -e
cd \$HOME
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --branch ${BRANCH} ${REPO_URL} ${REPO_DIR}
fi
cd \$HOME/${REPO_DIR}
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
for i in \$(seq 1 30); do
  [ -f "${REMOTE_TARBALL}" ] && break
  echo "[init] 업로드된 데이터 번들 대기중... (\$i)"; sleep 2
done
TARBALL=${REMOTE_TARBALL} bash scripts/prepare_data_in_container.sh || \
  echo "[init][WARN] 데이터 배치 실패 — 접속 후 수동으로 실행하세요: TARBALL=${REMOTE_TARBALL} bash scripts/prepare_data_in_container.sh"
echo "[init] 셋업 완료. 학습 실행: cd \$HOME/${REPO_DIR} && bash scripts/run_train.sh"
INIT
)

echo "[INFO] 워크스페이스 생성: ${WS_NAME}  (cluster=${CLUSTER}, resource=${RESOURCE})"
echo "[INFO] repo=${REPO_URL} (branch=${BRANCH})"
echo "[INFO] 데이터 업로드: ${TARBALL} → ${REMOTE_TARBALL}"

vessl workspace create "${WS_NAME}" \
  --cluster "${CLUSTER}" \
  --resource "${RESOURCE}" \
  --image "${IMAGE}" \
  --max-hours "${MAX_HOURS}" \
  --upload-local-file "${TARBALL}:${REMOTE_TARBALL}" \
  --init-script "${INIT_SCRIPT}"

echo
echo "[INFO] 생성 요청 완료. 접속 방법:"
echo "  vessl workspace list"
echo "  vessl workspace ssh          # SSH 접속"
echo "  # 또는 VESSL 웹 UI 에서 Jupyter/VSCode"
echo "  cd ~/${REPO_DIR} && bash scripts/run_train.sh   # 학습 실행"

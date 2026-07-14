#!/usr/bin/env bash
set -euo pipefail

# 옵션2(런타임 데이터 전송) 방식으로 VESSL 인터랙티브 워크스페이스를 띄운다.
#   - 코드: 공개 GitHub repo 에서 git clone (init-script 가 자동 수행 + pip install)
#   - 데이터: 최소 LMDB 번들(diary-lmdb.tar.gz)을 워크스페이스에만 반입
#            (영구 공용 Dataset 에 남기지 않음 → 프라이버시 보호)
#
# NOTE: 이 snu-eng-dgx 클러스터에서는 `vessl workspace create --upload-local-file`
#       이 서버 500 을 내고, 노드 SSH 포트로의 직접 scp 도 막혀 있다. 그래서
#       데이터는 워크스페이스 생성 후 웹 UI(Jupyter/VSCode)로 업로드한다(768K, 수 초).
#
# 사용 예:
#   bash scripts/run_workspace.sh
#   → 출력된 워크스페이스 URL 접속 → Jupyter/VSCode 로 diary-lmdb.tar.gz 업로드
#   → 터미널에서:
#       TARBALL=~/diary-lmdb.tar.gz bash ~/LLM-for-personality/scripts/prepare_data_in_container.sh
#       cd ~/LLM-for-personality && bash scripts/run_train.sh

cd "$(dirname "$0")/.."

REPO_URL="${REPO_URL:-https://github.com/bjh18055/LLM-for-personality.git}"
REPO_DIR="${REPO_DIR:-LLM-for-personality}"
BRANCH="${BRANCH:-main}"

CLUSTER="${CLUSTER:-snu-eng-dgx}"
RESOURCE="${RESOURCE:-a100-1}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.05-py3}"
# 공용 조직 목록에 노출돼도 내용을 짐작할 수 없도록 밋밋한 이름 사용.
WS_NAME="${WS_NAME:-dev-$(date +%m%d-%H%M)}"
MAX_HOURS="${MAX_HOURS:-6}"

# --- 사전 점검 ---
if ! command -v vessl >/dev/null 2>&1; then
  echo "[ERROR] 'vessl' 명령을 찾을 수 없습니다. VESSL 환경을 활성화하세요."
  exit 1
fi
if ! vessl whoami >/dev/null 2>&1; then
  echo "[ERROR] VESSL 인증이 안 되어 있습니다. 'vessl configure' 실행 후 재시도."
  exit 1
fi

# --- 업로드할 데이터 번들을 로컬에 준비 (최소 LMDB) ---
echo "[INFO] 데이터 번들 준비..."
bash scripts/pack_data.sh
echo "[INFO] 위 diary-lmdb.tar.gz 를 워크스페이스 웹 UI 로 업로드하세요 (아래 안내 참고)."

# --- init-script: 컨테이너 시작 시 코드 clone + 의존성 설치 ---
INIT_SCRIPT=$(cat <<INIT
set -e
cd \$HOME
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --branch ${BRANCH} ${REPO_URL} ${REPO_DIR}
fi
cd \$HOME/${REPO_DIR}
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "[init] 코드/의존성 준비 완료."
echo "[init] 이제 diary-lmdb.tar.gz 를 업로드한 뒤 아래를 실행하세요:"
echo "[init]   TARBALL=\\\$HOME/diary-lmdb.tar.gz bash \\\$HOME/${REPO_DIR}/scripts/prepare_data_in_container.sh"
echo "[init]   cd \\\$HOME/${REPO_DIR} && bash scripts/run_train.sh"
INIT
)

echo "[INFO] 워크스페이스 생성: ${WS_NAME}  (cluster=${CLUSTER}, resource=${RESOURCE})"
echo "[INFO] repo=${REPO_URL} (branch=${BRANCH})"

vessl workspace create "${WS_NAME}" \
  --cluster "${CLUSTER}" \
  --resource "${RESOURCE}" \
  --image-url "${IMAGE}" \
  --max-hours "${MAX_HOURS}" \
  --init-script "${INIT_SCRIPT}"

cat <<'GUIDE'

──────────────────────────────────────────────────────────────
다음 단계:
 1) 위 출력의 워크스페이스 URL 을 브라우저로 연다 (또는 `vessl workspace list`).
 2) Status 가 running 이 되면 Jupyter(또는 VSCode)를 연다.
 3) diary-lmdb.tar.gz 를 홈(~/)에 드래그해서 업로드한다 (768K, 수 초).
 4) 워크스페이스 터미널에서 실행:
      TARBALL=~/diary-lmdb.tar.gz bash ~/LLM-for-personality/scripts/prepare_data_in_container.sh
      cd ~/LLM-for-personality && bash scripts/run_train.sh
 * 대화형 SSH 접속: `vessl workspace ssh`
──────────────────────────────────────────────────────────────
GUIDE

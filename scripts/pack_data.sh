#!/usr/bin/env bash
set -euo pipefail

# 학습에 실제로 필요한 최소 LMDB 파일만 tar 로 묶는다.
# 인스타그램 원본 미디어(수백 MB)는 제외하고, 일기 텍스트가 담긴
# per-account LMDB(data/instagram-*/data.lmdb) + merged 캐시(cache/*.lmdb)의
# data.mdb 만 포함한다. 결과물은 보통 수 MB 로, VESSL 워크스페이스에
# 런타임 업로드해도 몇 초면 끝난다 (영구 공용 저장소에 남기지 않음).

cd "$(dirname "$0")/.."

OUT="${OUT:-diary-lmdb.tar.gz}"

FILES=()
while IFS= read -r f; do
  [ -n "$f" ] && FILES+=("$f")
done < <(
  find data -type f -path '*/data.lmdb/data.mdb' 2>/dev/null
  find cache -type f -path '*.lmdb/data.mdb' 2>/dev/null
)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "[ERROR] LMDB 파일을 찾지 못했습니다. 먼저 로컬에서 캐시를 빌드하세요:"
  echo "        python dataset.py   # 또는 학습을 한 번 로컬 실행"
  exit 1
fi

tar -czf "$OUT" "${FILES[@]}"

echo "[INFO] 생성됨: $OUT"
du -sh "$OUT"
echo "[INFO] 포함된 파일:"
tar -tzf "$OUT"

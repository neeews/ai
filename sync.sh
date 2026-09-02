#!/usr/bin/env bash
# 로컬 코드를 학습 서버로 올린다. 데이터·모델·venv는 서버에만 두고 건드리지 않는다.
#
# 접속 정보는 저장소에 두지 않는다. 옆에 .deploy.env 를 만들어 두면 자동으로 읽는다:
#   REMOTE=사용자@호스트
#   PORT=22
#   REMOTE_DIR=~/ai
set -euo pipefail
cd "$(dirname "$0")"

[ -f .deploy.env ] && . ./.deploy.env

REMOTE="${REMOTE:-}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-~/ai}"

if [ -z "$REMOTE" ]; then
  echo "REMOTE 이 설정되지 않았습니다. .deploy.env 를 만들거나 REMOTE=user@host 로 지정하세요." >&2
  exit 1
fi

rsync -avz --delete \
  --exclude venv/ --exclude data/ --exclude models/ --exclude __pycache__/ \
  --exclude '*.log' --exclude .git/ --exclude .deploy.env \
  -e "ssh -p ${PORT}" \
  ./ "${REMOTE}:${REMOTE_DIR}/"
echo "동기화 완료 → ${REMOTE}:${REMOTE_DIR}/"

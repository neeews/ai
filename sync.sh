#!/usr/bin/env bash
# 로컬 코드를 학습 서버로 올린다. 데이터·모델·venv는 서버에만 두고 건드리지 않는다.
#
# 접속 정보는 저장소에 두지 않는다. 옆에 .deploy.env 를 만들어 두면 자동으로 읽는다:
#   REMOTE=사용자@호스트
#   PORT=22
#   REMOTE_DIR='~/ai'   # 따옴표 필수. 안 묶으면 bash 가 ~ 를 로컬 홈으로 펼친다
#   SSHPASS=비밀번호   # (선택) 키 대신 비밀번호로 붙을 때. sshpass 가 설치돼 있어야 한다.
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

# SSHPASS 가 설정돼 있으면 sshpass 로 감싼다. -e 로 넘겨야 ps 목록에 비밀번호가 안 뜬다.
SSH_CMD="ssh -p ${PORT}"
if [ -n "${SSHPASS:-}" ]; then
  command -v sshpass >/dev/null || { echo "sshpass 가 없습니다. sudo apt install sshpass" >&2; exit 1; }
  export SSHPASS
  SSH_CMD="sshpass -e ${SSH_CMD}"
fi

rsync -avz --delete \
  --exclude venv/ --exclude data/ --exclude models/ --exclude __pycache__/ \
  --exclude '*.log' --exclude .git/ --exclude .deploy.env \
  -e "${SSH_CMD}" \
  ./ "${REMOTE}:${REMOTE_DIR}/"
echo "동기화 완료 → ${REMOTE}:${REMOTE_DIR}/"

"""neeews MySQL 접속 헬퍼.

접속 정보는 백엔드가 쓰는 ~/backend/.env 를 그대로 읽는다 (비밀번호를 이 저장소에 두지 않기 위함).
필요하면 환경변수 DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME 로 덮어쓸 수 있다.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pymysql

BACKEND_ENV = Path.home() / "backend" / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE 형식만 골라 읽는다. 따옴표와 주석은 벗겨낸다."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if raw and raw[0] in "\"'" and raw[-1] == raw[0] and len(raw) > 1:
            raw = raw[1:-1]
        values[key] = raw
    return values


def connect() -> pymysql.connections.Connection:
    env = _read_env_file(BACKEND_ENV)

    def get(key: str, default: str = "") -> str:
        return os.environ.get(key) or env.get(key) or default

    return pymysql.connect(
        host=get("DB_HOST", "127.0.0.1"),
        port=int(get("DB_PORT", "3306")),
        user=get("DB_USER"),
        password=get("DB_PASSWORD"),
        database=get("DB_NAME", "neeews"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

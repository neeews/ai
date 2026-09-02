"""DB의 articles 에서 라벨링할 기사를 뽑아 data/raw/articles.jsonl 에 저장한다.

카테고리별로 고르게 뽑아야 중요도 분포가 한쪽으로 쏠리지 않는다.
이미 뽑아둔 기사는 건너뛰므로 여러 번 실행해 풀을 늘려도 된다.

    python3 export_db.py --n 3000               # 카테고리 균등 3000건
    python3 export_db.py --n 500 --min-len 300  # 본문 300자 이상만
    python3 export_db.py --n 200 --category 정치
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from config import RAW_PATH
from db import connect

# 라벨링 가치가 없는 기사를 걸러내는 조건
BASE_WHERE = "title IS NOT NULL AND CHAR_LENGTH(title) >= 5"


def load_existing_ids() -> set[int]:
    if not RAW_PATH.exists():
        return set()
    ids = set()
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(int(json.loads(line)["id"]))
    return ids


def fetch_categories(cur) -> list[str]:
    cur.execute("SELECT DISTINCT category FROM articles WHERE category IS NOT NULL")
    return [row["category"] for row in cur.fetchall()]


def fetch_sample(cur, category: str | None, limit: int, min_len: int, exclude: set[int]) -> list[dict]:
    """무작위 표본. 최신 기사만 뽑히면 시기 편향이 생기므로 RAND() 로 섞는다."""
    where = [BASE_WHERE, "CHAR_LENGTH(COALESCE(description, '')) >= %s"]
    params: list = [min_len]
    if category:
        where.append("category = %s")
        params.append(category)

    # 제외 목록이 크면 SQL에 다 넣기보다 넉넉히 받아서 파이썬에서 거른다
    cur.execute(
        f"""SELECT id, title, description, category, source, published_at, link
            FROM articles
            WHERE {' AND '.join(where)}
            ORDER BY RAND()
            LIMIT %s""",
        (*params, limit * 2 if exclude else limit),
    )
    rows = [r for r in cur.fetchall() if r["id"] not in exclude]
    return rows[:limit]


def to_record(row: dict) -> dict:
    return {
        "id": int(row["id"]),
        "title": (row["title"] or "").strip(),
        "body": (row["description"] or "").strip(),
        "category": row["category"] or "",
        "source": row["source"] or "",
        "published_at": row["published_at"].isoformat() if row["published_at"] else "",
        "link": row["link"] or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DB에서 라벨링용 기사 추출")
    parser.add_argument("--n", type=int, default=1000, help="가져올 총 기사 수 (기본 1000)")
    parser.add_argument("--min-len", type=int, default=100,
                        help="본문 최소 글자 수. 너무 짧으면 중요도 판단이 어렵다 (기본 100)")
    parser.add_argument("--category", help="특정 카테고리만")
    parser.add_argument("--seed", type=int, default=None, help="MySQL RAND() 시드 고정")
    args = parser.parse_args()

    existing = load_existing_ids()
    print(f"기존 기사 풀: {len(existing)}건")

    conn = connect()
    try:
        with conn.cursor() as cur:
            if args.seed is not None:
                cur.execute("SELECT RAND(%s)", (args.seed,))

            if args.category:
                targets = {args.category: args.n}
            else:
                cats = fetch_categories(cur)
                per = max(1, args.n // len(cats))
                targets = {c: per for c in cats}

            collected: list[dict] = []
            for category, quota in targets.items():
                rows = fetch_sample(cur, category, quota, args.min_len, existing)
                print(f"  {category:<10} {len(rows):>5}건")
                collected.extend(to_record(r) for r in rows)
    finally:
        conn.close()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("a", encoding="utf-8") as f:
        for rec in collected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = Counter(r["category"] for r in collected)
    print(f"\n새로 추가 {len(collected)}건 → {RAW_PATH}")
    print(f"전체 풀: {len(existing) + len(collected)}건")
    print("카테고리 분포: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))


if __name__ == "__main__":
    main()

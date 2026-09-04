"""DB의 articles 에서 라벨링할 기사를 뽑아 data/raw/articles.jsonl 에 저장한다.

카테고리별로 고르게, 그리고 **날짜별로도 고르게** 뽑는다. 둘 중 하나라도 쏠리면
모델이 본문이 아니라 카테고리나 그날의 큰 사건을 외운다.
이미 뽑아둔 기사는 건너뛰므로 여러 번 실행해 풀을 늘려도 된다.

    python3 export_db.py --n 3000                # 카테고리·날짜 균등 3000건
    python3 export_db.py --n 150 --fresh         # AI가 판단한 적 없는 기사만
    python3 export_db.py --n 200 --category 정치

`--min-len` 기본값이 0인 이유: description 은 전문이 아니라 RSS 요약이라 평균 244자뿐이고,
길이 필터는 사실상 매체 필터로 작동한다 (300자 이상 통과율이 ZDNET 98% / 연합 0.3%).
300 을 걸면 IT/과학이 57%를 차지해 카테고리 균등이 깨진다. 제목만으로도 판정이 되므로
(제목만 52% vs 제목+본문 48%) 기본은 필터를 걸지 않는다.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from autolabel import EXAMPLE_IDS
from config import AUTO_LABEL_PATH, BACKEND_LABEL_PATH, RAW_PATH
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


def load_ai_judged_ids() -> set[int]:
    """LLM 이 이미 등급을 매긴 기사 + 프롬프트 예시. '한 번도 안 본 기사'를 뽑을 때 제외한다."""
    ids = set(EXAMPLE_IDS)
    for path in (AUTO_LABEL_PATH, BACKEND_LABEL_PATH):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(int(json.loads(line)["id"]))
    return ids


def spread_by_day(rows: list[dict], limit: int) -> list[dict]:
    """날짜별로 돌아가며 한 건씩 집어 특정 날짜에 쏠리지 않게 한다.

    하루치만 뽑으면 그날의 큰 사건이 '중요'의 정의가 되어 모델이 거기에 과적합된다.
    """
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        day = row["published_at"].date().isoformat() if row["published_at"] else ""
        by_day[day].append(row)

    picked: list[dict] = []
    days = sorted(by_day)
    while len(picked) < limit and any(by_day[d] for d in days):
        for day in days:
            if by_day[day]:
                picked.append(by_day[day].pop())
                if len(picked) >= limit:
                    break
    return picked


def fetch_sample(cur, category: str | None, limit: int, min_len: int, exclude: set[int]) -> list[dict]:
    """무작위 표본을 넉넉히 받아 제외 목록을 거른 뒤, 날짜가 고르게 퍼지도록 고른다."""
    where = [BASE_WHERE, "CHAR_LENGTH(COALESCE(description, '')) >= %s"]
    params: list = [min_len]
    if category:
        where.append("category = %s")
        params.append(category)

    # 날짜를 고르게 퍼뜨리려면 후보가 넉넉해야 한다. 제외 목록까지 감안해 넉넉히 받는다.
    cur.execute(
        f"""SELECT id, title, description, category, source, published_at, link
            FROM articles
            WHERE {' AND '.join(where)}
            ORDER BY RAND()
            LIMIT %s""",
        (*params, max(limit * 10, 200)),
    )
    rows = [r for r in cur.fetchall() if r["id"] not in exclude]
    return spread_by_day(rows, limit)


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
    parser.add_argument("--min-len", type=int, default=0,
                        help="본문 최소 글자 수. 0 이 기본 — 길이 필터는 사실상 매체 필터다 (docstring 참고)")
    parser.add_argument("--category", help="특정 카테고리만")
    parser.add_argument("--fresh", action="store_true",
                        help="LLM 이 아직 판단한 적 없는 기사만 (자동라벨·백엔드시드·프롬프트 예시 제외)")
    parser.add_argument("--seed", type=int, default=None, help="MySQL RAND() 시드 고정")
    args = parser.parse_args()

    pool_ids = load_existing_ids()
    print(f"기존 기사 풀: {len(pool_ids)}건")
    # 뽑기에서 제외할 id (풀에 이미 있는 것 + --fresh 면 LLM 이 이미 본 것)
    exclude = set(pool_ids)
    if args.fresh:
        ai_judged = load_ai_judged_ids()
        exclude |= ai_judged
        print(f"--fresh: LLM 이 이미 본 기사 {len(ai_judged)}건도 제외 "
              f"(프롬프트 예시 {len(EXAMPLE_IDS)}건 포함)")

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
                rows = fetch_sample(cur, category, quota, args.min_len, exclude)
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
    print(f"전체 풀: {len(pool_ids) + len(collected)}건")
    print("카테고리 분포: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))

    days = Counter(r["published_at"][:10] for r in collected if r["published_at"])
    if days:
        top_day, top_n = days.most_common(1)[0]
        print(f"날짜 분포: {len(days)}일에 걸침 (하루 최대 {top_n}건 — {top_day})")
        if top_n > max(3, len(collected) * 0.2):
            print(f"  ⚠ '{top_day}' 하루가 {top_n / len(collected):.0%}를 차지합니다. "
                  f"그날의 큰 사건에 모델이 과적합됩니다.")


if __name__ == "__main__":
    main()

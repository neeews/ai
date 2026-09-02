"""백엔드가 만들어 둔 중요도 라벨(LOW/MEDIUM/HIGH)을 학습용 라벨로 가져온다.

라벨만 있고 본문이 없으므로, 기사 id로 DB에서 제목·본문을 채워 넣는다.
가져온 라벨은 사람이 매긴 정답지(data/labeled/labels.jsonl)로 취급한다.

    python3 import_labels.py --csv labels/importance_seed.csv
    python3 import_labels.py --from-db          # article_importance_labels 테이블에서 직접
    python3 import_labels.py --csv seed.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import BACKEND_TO_LABEL, LABEL_NAMES, LABELED_PATH, RAW_PATH


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> list[tuple[int, str]]:
    """article_id, label 두 컬럼만 쓴다. 나머지 컬럼은 있어도 무시."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            aid = row.get("article_id") or row.get("id")
            label = (row.get("label") or "").strip().upper()
            if aid and label:
                rows.append((int(aid), label))
        return rows


def read_db_labels() -> list[tuple[int, str]]:
    from db import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT article_id, label FROM article_importance_labels")
            return [(int(r["article_id"]), str(r["label"]).upper()) for r in cur.fetchall()]
    finally:
        conn.close()


def fetch_articles(ids: list[int]) -> dict[int, dict]:
    """기사 본문을 DB에서 채운다. id 목록이 길 수 있으므로 나눠서 조회한다."""
    from db import connect
    found: dict[int, dict] = {}
    conn = connect()
    try:
        with conn.cursor() as cur:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                placeholders = ",".join(["%s"] * len(chunk))
                cur.execute(
                    f"""SELECT id, title, description, category, source, published_at, link
                        FROM articles WHERE id IN ({placeholders})""",
                    chunk,
                )
                for r in cur.fetchall():
                    found[int(r["id"])] = {
                        "id": int(r["id"]),
                        "title": (r["title"] or "").strip(),
                        "body": (r["description"] or "").strip(),
                        "category": r["category"] or "",
                        "source": r["source"] or "",
                        "published_at": r["published_at"].isoformat() if r["published_at"] else "",
                        "link": r["link"] or "",
                    }
    finally:
        conn.close()
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="백엔드 중요도 라벨 가져오기")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="importance_seed.csv 경로")
    src.add_argument("--from-db", action="store_true", help="article_importance_labels 테이블에서")
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 확인")
    args = parser.parse_args()

    pairs = read_csv(args.csv) if args.csv else read_db_labels()
    print(f"라벨 {len(pairs)}건 읽음")

    unknown = {lbl for _, lbl in pairs if lbl not in BACKEND_TO_LABEL}
    if unknown:
        raise SystemExit(
            f"모르는 라벨 값: {sorted(unknown)}\n"
            f"config.py 의 BACKEND_TO_LABEL 에 매핑을 추가하세요. (현재: {BACKEND_TO_LABEL})"
        )

    articles = fetch_articles([aid for aid, _ in pairs])
    missing = [aid for aid, _ in pairs if aid not in articles]
    if missing:
        print(f"  ⚠ DB에 없는 기사 {len(missing)}건 제외: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # 기사 풀(raw)에도 없으면 라벨만 있고 학습에 못 쓰므로 함께 채워 넣는다
    pool = {int(a["id"]): a for a in load_jsonl(RAW_PATH)}
    new_pool = [a for aid, a in articles.items() if aid not in pool]

    existing = {int(r["id"]) for r in load_jsonl(LABELED_PATH)}
    rows, skipped = [], 0
    counts: Counter[int] = Counter()
    for aid, backend_label in pairs:
        if aid not in articles:
            continue
        if aid in existing:
            skipped += 1
            continue
        label = BACKEND_TO_LABEL[backend_label]
        counts[label] += 1
        rows.append({
            "id": aid,
            "label": label,
            "title": articles[aid]["title"],
            "category": articles[aid]["category"],
            "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "origin": "backend",
        })

    print(f"\n가져올 라벨 {len(rows)}건" + (f" (이미 있어 건너뜀 {skipped}건)" if skipped else ""))
    for lv in sorted(LABEL_NAMES):
        print(f"  {lv} {LABEL_NAMES[lv]:<10} {counts.get(lv, 0):>4}건")

    cats = Counter(articles[aid]["category"] for aid, _ in pairs if aid in articles)
    print("\n카테고리 분포: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()))
    top_cat, top_n = cats.most_common(1)[0] if cats else ("", 0)
    if cats and top_n / sum(cats.values()) > 0.4:
        print(f"  ⚠ '{top_cat}'가 {top_n / sum(cats.values()):.0%}를 차지합니다. "
              f"카테고리가 한쪽으로 쏠리면 모델이 카테고리만 보고 중요도를 찍습니다.")

    if args.dry_run:
        print("\n--dry-run 이므로 저장하지 않았습니다.")
        return

    if new_pool:
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RAW_PATH.open("a", encoding="utf-8") as f:
            for a in new_pool:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"\n기사 풀에 {len(new_pool)}건 추가 → {RAW_PATH}")

    LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELED_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"라벨 {len(rows)}건 저장 → {LABELED_PATH}")


if __name__ == "__main__":
    main()

"""백엔드가 만들어 둔 중요도 라벨(LOW/MEDIUM/HIGH)을 학습용 라벨로 가져온다.

라벨만 있고 본문이 없으므로, 기사 id로 DB에서 제목·본문을 채워 넣는다.
가져온 라벨은 **학습용**(data/labeled/backend_labels.jsonl)으로 들어간다.
정답지(labels.jsonl)에는 절대 넣지 않는다 — 백엔드 시드는 사람이 아니라
claude-opus-5 가 매긴 것이라, 그걸로 채점하면 "모델이 opus 를 얼마나 흉내내나"만 재게 된다.

DB 테이블에 라벨 주체 컬럼이 없으므로 --labeled-by 로 누가 매겼는지 직접 적는다.
사람이 매긴 라벨을 반입할 때만 --labeled-by human 을 쓰고, 그 경우에도
정답지로 쓰려면 label.py 로 다시 매기는 편이 안전하다.

    python3 import_labels.py --csv labels/importance_seed.csv
    python3 import_labels.py --from-db          # article_importance_labels 테이블에서 직접
    python3 import_labels.py --csv seed.csv --dry-run
    python3 import_labels.py --migrate          # 예전 labels.jsonl 에 섞여 들어간 라벨 분리
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import (
    BACKEND_LABEL_PATH,
    BACKEND_LABELED_BY,
    BACKEND_TO_LABEL,
    HUMAN_ORIGIN,
    LABEL_NAMES,
    LABELED_PATH,
    RAW_PATH,
)


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def migrate(labeled_by: str, dry_run: bool) -> None:
    """예전 labels.jsonl 에 섞여 들어간 비-사람 라벨을 backend_labels.jsonl 로 옮긴다.

    origin 이 없는 행은 출처를 확인할 방법이 없으므로 unknown 으로 찍어 남긴다.
    정답지 자격은 '사람이 매겼음이 기록된 것'뿐이고, 모르면 자격이 없다.
    """
    rows = load_jsonl(LABELED_PATH)
    if not rows:
        print(f"옮길 라벨이 없습니다. ({LABELED_PATH})")
        return

    keep, moved, unknown = [], [], 0
    for row in rows:
        origin = row.get("origin")
        if origin == HUMAN_ORIGIN:
            keep.append(row)
        elif origin is None:
            row["origin"] = "unknown"
            unknown += 1
            keep.append(row)
        else:
            row["origin"] = labeled_by if origin == "backend" else origin
            moved.append(row)

    print(f"정답지 {len(rows)}건 검사")
    print(f"  사람 라벨로 남김      {len(keep) - unknown:>4}건")
    print(f"  출처 불명(unknown)   {unknown:>4}건" + ("   ⚠ 정답지에서 제외됩니다" if unknown else ""))
    print(f"  학습용으로 이동       {len(moved):>4}건 → {BACKEND_LABEL_PATH.name} (origin={labeled_by})")
    if unknown:
        print("\n  출처 불명 행은 labels.jsonl 에 남지만 test 에는 들어가지 않습니다.")
        print("  사람이 매긴 것이 확실하다면 그 행의 origin 을 \"human\" 으로 직접 고치세요.")
    if dry_run:
        print("\n--dry-run 이므로 저장하지 않았습니다.")
        return

    existing = {int(r["id"]) for r in load_jsonl(BACKEND_LABEL_PATH)}
    new_rows = [r for r in moved if int(r["id"]) not in existing]
    append_jsonl(BACKEND_LABEL_PATH, new_rows)
    write_jsonl(LABELED_PATH, keep)
    print(f"\n저장 완료: {LABELED_PATH} {len(keep)}건 / {BACKEND_LABEL_PATH} +{len(new_rows)}건")


def read_csv(path: Path) -> list[tuple[int, str, str | None]]:
    """article_id, label 두 컬럼만 쓴다. 나머지 컬럼은 있어도 무시."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            aid = row.get("article_id") or row.get("id")
            label = (row.get("label") or "").strip().upper()
            if aid and label:
                rows.append((int(aid), label, None))
        return rows


def read_db_labels() -> list[tuple[int, str, str | None]]:
    """(기사id, 라벨, 매긴 주체) 를 읽는다. 주체를 모르면 None 이고, 그러면 --labeled-by 값을 쓴다.

    출처 컬럼은 migration_20260903_label_provenance.sql 로 추가된다.
    아직 적용 전인 DB에서도 돌아가야 하므로 컬럼이 없으면 옛 쿼리로 물러선다.
    """
    from db import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT article_id, label, labeled_by, origin FROM article_importance_labels "
                    "WHERE round_no = 0"
                )
                rows = cur.fetchall()
                return [
                    (int(r["article_id"]), str(r["label"]).upper(),
                     HUMAN_ORIGIN if str(r["origin"]).upper() == "HUMAN" else str(r["labeled_by"]))
                    for r in rows
                ]
            except Exception:
                conn.rollback()
                print("  ⚠ 출처 컬럼이 없는 DB입니다. --labeled-by 값으로 일괄 기록합니다.")
                print("    (labels/migration_20260903_label_provenance.sql 을 적용하면 자동으로 구분됩니다)")
            cur.execute("SELECT article_id, label FROM article_importance_labels")
            return [(int(r["article_id"]), str(r["label"]).upper(), None) for r in cur.fetchall()]
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
    src.add_argument("--migrate", action="store_true",
                     help="예전 labels.jsonl 에 섞인 비-사람 라벨을 학습용 파일로 분리")
    parser.add_argument("--labeled-by", default=BACKEND_LABELED_BY,
                        help=f"이 라벨을 실제로 매긴 주체 (기본: {BACKEND_LABELED_BY})")
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 확인")
    args = parser.parse_args()

    if args.labeled_by == HUMAN_ORIGIN:
        print("⚠ --labeled-by human: 이 라벨은 정답지 자격을 갖습니다. 정말 사람이 매긴 게 맞습니까?")

    if args.migrate:
        migrate(args.labeled_by, args.dry_run)
        return

    pairs = read_csv(args.csv) if args.csv else read_db_labels()
    print(f"라벨 {len(pairs)}건 읽음")

    unknown = {lbl for _, lbl, _ in pairs if lbl not in BACKEND_TO_LABEL}
    if unknown:
        raise SystemExit(
            f"모르는 라벨 값: {sorted(unknown)}\n"
            f"config.py 의 BACKEND_TO_LABEL 에 매핑을 추가하세요. (현재: {BACKEND_TO_LABEL})"
        )

    articles = fetch_articles([aid for aid, _, _ in pairs])
    missing = [aid for aid, _, _ in pairs if aid not in articles]
    if missing:
        print(f"  ⚠ DB에 없는 기사 {len(missing)}건 제외: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # 기사 풀(raw)에도 없으면 라벨만 있고 학습에 못 쓰므로 함께 채워 넣는다
    pool = {int(a["id"]): a for a in load_jsonl(RAW_PATH)}
    new_pool = [a for aid, a in articles.items() if aid not in pool]

    # 같은 기사에 사람 라벨과 AI 라벨이 둘 다 있을 수 있다 (채점용으로 일부러 겹쳐 매긴 경우).
    # 중복은 **목적지 파일별로** 따져야 한다. id 하나당 한 행만 받으면 먼저 온 AI 라벨이 이기고
    # 사람 라벨이 조용히 사라진다 — 정답지가 통째로 없어지는 사고가 난다.
    existing_by_dest = {
        HUMAN_ORIGIN: {int(r["id"]) for r in load_jsonl(LABELED_PATH)},
        "other": {int(r["id"]) for r in load_jsonl(BACKEND_LABEL_PATH)},
    }
    rows, skipped = [], 0
    counts: Counter[int] = Counter()
    for aid, backend_label, db_origin in pairs:
        if aid not in articles:
            continue
        origin = db_origin or args.labeled_by
        dest = HUMAN_ORIGIN if origin == HUMAN_ORIGIN else "other"
        if aid in existing_by_dest[dest]:
            skipped += 1
            continue
        existing_by_dest[dest].add(aid)
        label = BACKEND_TO_LABEL[backend_label]
        counts[label] += 1
        rows.append({
            "id": aid,
            "label": label,
            "title": articles[aid]["title"],
            "category": articles[aid]["category"],
            "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "origin": origin,
        })

    print(f"\n가져올 라벨 {len(rows)}건" + (f" (이미 있어 건너뜀 {skipped}건)" if skipped else ""))
    for lv in sorted(LABEL_NAMES):
        print(f"  {lv} {LABEL_NAMES[lv]:<10} {counts.get(lv, 0):>4}건")

    cats = Counter(articles[aid]["category"] for aid, _, _ in pairs if aid in articles)
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

    # 출처가 사람인 행만 정답지로, 나머지는 학습용으로 간다. DB가 출처를 알려주면 그걸 따른다.
    human_rows = [r for r in rows if r["origin"] == HUMAN_ORIGIN]
    other_rows = [r for r in rows if r["origin"] != HUMAN_ORIGIN]
    if human_rows:
        append_jsonl(LABELED_PATH, human_rows)
        print(f"사람 라벨 {len(human_rows)}건 저장 → {LABELED_PATH}")
    if other_rows:
        append_jsonl(BACKEND_LABEL_PATH, other_rows)
        origins = ", ".join(sorted({r["origin"] for r in other_rows}))
        print(f"학습용 라벨 {len(other_rows)}건 저장 → {BACKEND_LABEL_PATH}  (origin={origins})")


if __name__ == "__main__":
    main()

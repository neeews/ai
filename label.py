"""기사에 중요도 0~4 라벨을 직접 매기는 CLI 도구.

여기서 만든 라벨은 **모델 성능을 재는 정답지(gold)** 로 쓰인다.
한 건 매길 때마다 즉시 저장되므로 Ctrl+C 로 끊고 나중에 이어서 해도 된다.

    python3 label.py                # 무작위 순서로 라벨링
    python3 label.py --review       # LLM이 매긴 라벨을 보고 검수(맞으면 Enter)
    python3 label.py --stats        # 분포만 확인
    python3 label.py --category 정치

키 입력:
    0~4  중요도 부여      s  건너뛰기      u  직전 라벨 취소
    g    라벨 기준 보기    q  종료
    (--review 모드에서는 Enter = LLM 라벨 그대로 인정)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone

from config import AUTO_LABEL_PATH, LABEL_NAMES, LABELED_PATH, RAW_PATH, ROOT, SEED


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_label(row: dict) -> None:
    LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELED_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(labels: list[dict]) -> None:
    if not labels:
        print("아직 라벨이 없습니다.")
        return
    counts = Counter(row["label"] for row in labels)
    total = len(labels)
    print(f"\n라벨 분포 (총 {total}건)")
    for lv in range(5):
        n = counts.get(lv, 0)
        bar = "█" * int(n / max(total, 1) * 40)
        print(f"  {lv} {LABEL_NAMES[lv]:<10} {n:>5}건 {bar}")
    weak = [lv for lv in range(5) if counts.get(lv, 0) < 30]
    if weak:
        print(f"\n  ⚠ 30건 미만 등급: {weak} — 이 등급은 평가 신뢰도가 낮습니다.")


def show_article(article: dict, idx: int, total: int, done: int, suggested: int | None) -> None:
    print("\n" + "=" * 78)
    head = f"[{idx}/{total}]  완료 {done}건  |  {article.get('category', '')}  |  {article.get('source', '')}"
    if article.get("published_at"):
        head += f"  |  {article['published_at'][:10]}"
    print(head)
    print("=" * 78)
    print(f"\n제목: {article['title']}\n")

    body = (article.get("body") or "").strip()
    if body:
        # 문단이 뭉쳐 있어도 읽히도록 적당히 끊어 출력
        text = body[:1200]
        for i in range(0, len(text), 100):
            print(f"  {text[i:i + 100]}")
        if len(body) > 1200:
            print("  ...")
    else:
        print("  (본문 없음 — 제목만 보고 판단하거나 s 로 건너뛰세요)")
    print(f"\n  링크: {article.get('link', '')}")
    if suggested is not None:
        print(f"  LLM 제안: {suggested} ({LABEL_NAMES[suggested]})   ← 맞으면 Enter")
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="기사 중요도 라벨링")
    parser.add_argument("--review", action="store_true",
                        help="LLM 자동 라벨을 제안으로 띄우고 검수 (Enter로 인정)")
    parser.add_argument("--stats", action="store_true", help="라벨 분포만 출력")
    parser.add_argument("--category", help="특정 카테고리만")
    parser.add_argument("--limit", type=int, help="이번 세션에서 라벨할 최대 건수")
    parser.add_argument("--no-shuffle", action="store_true", help="파일 순서대로 (기본은 무작위)")
    args = parser.parse_args()

    labels = load_jsonl(LABELED_PATH)
    if args.stats:
        print_stats(labels)
        return

    articles = load_jsonl(RAW_PATH)
    if not articles:
        print(f"기사 풀이 비어 있습니다. 먼저 `python3 export_db.py` 를 실행하세요. ({RAW_PATH})")
        return

    suggestions: dict[int, int] = {}
    if args.review:
        suggestions = {int(r["id"]): int(r["label"]) for r in load_jsonl(AUTO_LABEL_PATH)}
        if not suggestions:
            print(f"자동 라벨이 없습니다. `python3 autolabel.py` 를 먼저 실행하세요. ({AUTO_LABEL_PATH})")
            return

    labeled_ids = {int(row["id"]) for row in labels}
    todo = [a for a in articles if int(a["id"]) not in labeled_ids]
    if args.category:
        todo = [a for a in todo if a.get("category") == args.category]
    if args.review:
        todo = [a for a in todo if int(a["id"]) in suggestions]

    if not todo:
        print("라벨할 기사가 없습니다.")
        print_stats(labels)
        return

    if not args.no_shuffle:
        random.Random(SEED).shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"\n라벨 대상 {len(todo)}건 (이미 완료 {len(labels)}건)")
    print("4 매우중요 / 3 높음 / 2 보통 / 1 낮음 / 0 무시   ·   s 건너뛰기  u 취소  g 기준  q 종료")

    done = len(labels)
    session_rows: list[dict] = []

    for idx, article in enumerate(todo, start=1):
        aid = int(article["id"])
        suggested = suggestions.get(aid) if args.review else None
        show_article(article, idx, len(todo), done, suggested)

        while True:
            prompt = "중요도 [0-4/s/u/g/q]" + ("(Enter=제안) > " if suggested is not None else " > ")
            try:
                key = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n중단합니다. 지금까지 라벨은 저장되어 있습니다.")
                print_stats(load_jsonl(LABELED_PATH))
                return

            if key == "" and suggested is not None:
                key = str(suggested)
            if key == "q":
                print("\n종료합니다.")
                print_stats(load_jsonl(LABELED_PATH))
                return
            if key == "s":
                break
            if key == "g":
                print("\n" + (ROOT / "LABELING_GUIDE.md").read_text(encoding="utf-8"))
                continue
            if key == "u":
                if not session_rows:
                    print("  이번 세션에 취소할 라벨이 없습니다.")
                    continue
                removed = session_rows.pop()
                remaining = [r for r in load_jsonl(LABELED_PATH) if int(r["id"]) != int(removed["id"])]
                write_jsonl(LABELED_PATH, remaining)
                done -= 1
                print(f"  취소됨: [{removed['label']}] {removed['title'][:40]}")
                continue
            if key in {"0", "1", "2", "3", "4"}:
                row = {
                    "id": aid,
                    "label": int(key),
                    "title": article["title"],
                    "category": article.get("category", ""),
                    "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                if suggested is not None:
                    row["llm_label"] = suggested
                append_label(row)
                session_rows.append(row)
                done += 1
                mark = ""
                if suggested is not None:
                    mark = "  (LLM 일치)" if int(key) == suggested else f"  (LLM {suggested} → 수정)"
                print(f"  ✓ {key} ({LABEL_NAMES[int(key)]}){mark}")
                break
            print("  0~4, s, u, g, q 중 하나를 입력하세요.")

    print("\n이번 세션 라벨링을 마쳤습니다.")
    print_stats(load_jsonl(LABELED_PATH))


if __name__ == "__main__":
    main()

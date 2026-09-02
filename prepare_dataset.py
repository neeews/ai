"""라벨을 train/val/test 로 나눠 data/splits/ 에 저장한다.

기본 동작(hybrid): LLM 자동 라벨로 train/val 을 만들고, **test 는 사람이 매긴 라벨만** 쓴다.
사람 라벨이 들어간 기사는 train/val 에서 빼내므로 평가가 오염되지 않는다.

    python3 prepare_dataset.py             # hybrid (자동 라벨이 있으면 자동 선택)
    python3 prepare_dataset.py --mode human   # 사람 라벨만으로 학습·평가
    python3 prepare_dataset.py --mode auto    # 자동 라벨만 (검증용, 성능 수치는 못 믿음)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from config import (
    AUTO_LABEL_PATH,
    LABEL_NAMES,
    LABELED_PATH,
    RAW_PATH,
    SEED,
    SPLIT_DIR,
    TEST_RATIO,
    VAL_RATIO,
)


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_text(article: dict) -> str:
    """제목 + 본문. 토크나이저가 뒤를 잘라내므로 신호가 센 제목을 앞에 둔다."""
    title = (article.get("title") or "").strip()
    body = (article.get("body") or "").strip()
    return f"{title}\n{body}".strip()


def merge(articles: dict[int, dict], labels: list[dict]) -> list[dict]:
    rows, missing = [], 0
    for row in labels:
        article = articles.get(int(row["id"]))
        if article is None:
            missing += 1
            continue
        text = build_text(article)
        if not text:
            continue
        rows.append({
            "id": int(row["id"]),
            "label": int(row["label"]),
            "text": text,
            "title": article.get("title", ""),
            "category": article.get("category", ""),
        })
    if missing:
        print(f"  ⚠ 원본 기사를 못 찾은 라벨 {missing}건 제외")
    return rows


def stratified_split(rows: list[dict], ratios: tuple[float, float], seed: int):
    """등급별 비율을 유지하며 (val, test) 비율만큼 떼어낸다."""
    val_ratio, test_ratio = ratios
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for _, items in sorted(by_label.items()):
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio)) if n >= 5 else 0
        n_val = max(1, int(n * val_ratio)) if n >= 5 else 0
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def describe(name: str, rows: list[dict]) -> None:
    counts = Counter(r["label"] for r in rows)
    dist = "  ".join(f"{lv}:{counts.get(lv, 0)}" for lv in range(5))
    print(f"  {name:<6} {len(rows):>6}건   [{dist}]")


def agreement(human: list[dict], auto: list[dict]) -> None:
    """사람 라벨과 LLM 라벨이 겹치는 기사에서 일치율을 재 본다 — 자동 라벨 신뢰도의 근거."""
    auto_map = {int(r["id"]): int(r["label"]) for r in auto}
    pairs = [(int(r["label"]), auto_map[int(r["id"])]) for r in human if int(r["id"]) in auto_map]
    if not pairs:
        return
    exact = sum(1 for h, a in pairs if h == a) / len(pairs)
    close = sum(1 for h, a in pairs if abs(h - a) <= 1) / len(pairs)
    print(f"\nLLM 라벨 신뢰도 (겹치는 {len(pairs)}건 기준)")
    print(f"  정확 일치      {exact:.1%}")
    print(f"  ±1등급 이내    {close:.1%}")
    if exact < 0.4:
        print("  ⚠ 일치율이 낮습니다. autolabel.py 의 프롬프트를 손보거나 사람 라벨을 늘리세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="학습용 데이터셋 생성")
    parser.add_argument("--mode", choices=["hybrid", "human", "auto"], default="hybrid")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    articles = {int(a["id"]): a for a in load_jsonl(RAW_PATH)}
    human = load_jsonl(LABELED_PATH)
    auto = load_jsonl(AUTO_LABEL_PATH)

    mode = args.mode
    if mode == "hybrid" and not auto:
        print("자동 라벨이 없어 human 모드로 전환합니다.")
        mode = "human"
    if mode == "human" and not human:
        raise SystemExit(f"사람 라벨이 없습니다. `python3 label.py` 를 먼저 실행하세요. ({LABELED_PATH})")
    if mode == "auto" and not auto:
        raise SystemExit(f"자동 라벨이 없습니다. ({AUTO_LABEL_PATH})")

    if human and auto:
        agreement(human, auto)

    if mode == "hybrid":
        human_ids = {int(r["id"]) for r in human}
        # 사람이 본 기사는 학습에서 제외 → test 오염 방지
        auto_rows = merge(articles, [r for r in auto if int(r["id"]) not in human_ids])
        test = merge(articles, human)
        train, val, _ = stratified_split(auto_rows, (VAL_RATIO, 0.0), args.seed)
        print(f"\nhybrid 모드: train/val = LLM 라벨 {len(auto_rows)}건, test = 사람 라벨 {len(test)}건")
    else:
        rows = merge(articles, human if mode == "human" else auto)
        print(f"\n{mode} 모드: 총 {len(rows)}건")
        train, val, test = stratified_split(rows, (VAL_RATIO, TEST_RATIO), args.seed)

    all_rows = train + val + test
    counts = Counter(r["label"] for r in all_rows)
    print("\n전체 등급 분포")
    for lv in range(5):
        n = counts.get(lv, 0)
        print(f"  {lv} {LABEL_NAMES[lv]:<10} {n:>6}건" + ("   ⚠ 부족" if n < 20 else ""))
    if any(counts.get(lv, 0) == 0 for lv in range(5)):
        print("\n  ⚠ 데이터가 0건인 등급이 있습니다. 모델은 그 등급을 절대 예측하지 못합니다.")
    if not test:
        print("\n  ⚠ test 가 비었습니다. `python3 label.py` 로 평가용 라벨을 만드세요.")

    print("\n분할 결과")
    for name, split in (("train", train), ("val", val), ("test", test)):
        describe(name, split)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = SPLIT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  저장: {path}")


if __name__ == "__main__":
    main()

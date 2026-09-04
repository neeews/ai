"""라벨을 train/val/test 로 나눠 data/splits/ 에 저장한다.

기본 동작(hybrid): LLM 라벨(로컬 자동 라벨 + 백엔드 시드)로 train/val 을 만들고,
**test 는 origin="human" 인 라벨만** 쓴다. 사람이 본 기사는 train/val 에서 빼낸다.

test 자격을 파일이 아니라 행의 origin 으로 판정하는 이유: 예전에 백엔드 시드
(claude-opus-5 가 매긴 300건)가 labels.jsonl 로 그대로 들어가 정답지 행세를 했고,
그 위에서 잰 수치는 전부 "LLM 이 다른 LLM 을 얼마나 흉내내나"였다.

    python3 prepare_dataset.py             # hybrid (LLM 라벨이 있으면 자동 선택)
    python3 prepare_dataset.py --mode human   # 사람 라벨만으로 학습·평가
    python3 prepare_dataset.py --mode auto    # LLM 라벨만 (검증용, 성능 수치는 못 믿음)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from config import (
    AUTO_LABEL_PATH,
    BACKEND_LABEL_PATH,
    HUMAN_ORIGIN,
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
    dist = "  ".join(f"{lv}:{counts.get(lv, 0)}" for lv in sorted(LABEL_NAMES))
    print(f"  {name:<6} {len(rows):>6}건   [{dist}]")


def agreement(name: str, human: list[dict], other: list[dict]) -> None:
    """사람 라벨과 LLM 라벨이 겹치는 기사에서 일치율을 잰다 — LLM 라벨 신뢰도의 근거."""
    other_map = {int(r["id"]): int(r["label"]) for r in other}
    pairs = [(int(r["label"]), other_map[int(r["id"])]) for r in human if int(r["id"]) in other_map]
    if not pairs:
        print(f"\n{name} vs 사람: 겹치는 기사가 없어 신뢰도를 잴 수 없습니다.")
        return
    exact = sum(1 for h, a in pairs if h == a) / len(pairs)
    close = sum(1 for h, a in pairs if abs(h - a) <= 1) / len(pairs)
    print(f"\n{name} 라벨 신뢰도 (사람 라벨과 겹치는 {len(pairs)}건 기준)")
    print(f"  정확 일치      {exact:.1%}")
    print(f"  ±1등급 이내    {close:.1%}")
    if len(pairs) < 30:
        print(f"  ⚠ 겹치는 기사가 {len(pairs)}건뿐이라 이 수치는 흔들립니다.")
    if exact < 0.4:
        print("  ⚠ 일치율이 낮습니다. 프롬프트를 손보거나 사람 라벨을 늘리세요.")


def split_by_origin(rows: list[dict]) -> tuple[list[dict], Counter]:
    """정답지 자격이 있는 행(origin=human)만 골라낸다. 출처가 없으면 자격 없음."""
    origins = Counter(r.get("origin", "unknown") for r in rows if r.get("origin") != HUMAN_ORIGIN)
    return [r for r in rows if r.get("origin") == HUMAN_ORIGIN], origins


def main() -> None:
    parser = argparse.ArgumentParser(description="학습용 데이터셋 생성")
    parser.add_argument("--mode", choices=["hybrid", "human", "auto"], default="hybrid")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    articles = {int(a["id"]): a for a in load_jsonl(RAW_PATH)}
    labeled = load_jsonl(LABELED_PATH)
    auto = load_jsonl(AUTO_LABEL_PATH)
    backend = load_jsonl(BACKEND_LABEL_PATH)

    human, origins = split_by_origin(labeled)
    rejected = len(labeled) - len(human)
    if rejected:
        print(f"⚠ {LABELED_PATH.name} 의 {rejected}건은 origin 이 사람이 아니라 test 에서 제외합니다.")
        print("   출처별: " + ", ".join(f"{k} {v}건" for k, v in origins.most_common()))
        print("   `python3 import_labels.py --migrate` 로 학습용 파일에 옮기세요.\n")

    # 같은 기사에 자동 라벨과 시드 라벨이 다 있으면 시드(더 큰 모델)를 쓴다
    llm_map = {int(r["id"]): r for r in auto}
    llm_map.update({int(r["id"]): r for r in backend})
    llm = list(llm_map.values())

    mode = args.mode
    if mode == "hybrid" and not llm:
        print("LLM 라벨이 없어 human 모드로 전환합니다.")
        mode = "human"
    if mode == "human" and not human:
        raise SystemExit(f"사람 라벨이 없습니다. `python3 label.py` 를 먼저 실행하세요. ({LABELED_PATH})")
    if mode == "auto" and not llm:
        raise SystemExit(f"LLM 라벨이 없습니다. ({AUTO_LABEL_PATH}, {BACKEND_LABEL_PATH})")

    if human:
        if auto:
            agreement("자동(Ollama)", human, auto)
        if backend:
            agreement("백엔드 시드", human, backend)

    if mode == "hybrid":
        human_ids = {int(r["id"]) for r in human}
        # 사람이 본 기사는 학습에서 제외 → test 오염 방지
        llm_rows = merge(articles, [r for r in llm if int(r["id"]) not in human_ids])
        test = merge(articles, human)
        train, val, _ = stratified_split(llm_rows, (VAL_RATIO, 0.0), args.seed)
        print(f"\nhybrid 모드: train/val = LLM 라벨 {len(llm_rows)}건"
              f"(자동 {len(auto)} + 시드 {len(backend)}, 중복·사람겹침 제외), "
              f"test = 사람 라벨 {len(test)}건")
    else:
        rows = merge(articles, human if mode == "human" else llm)
        print(f"\n{mode} 모드: 총 {len(rows)}건")
        train, val, test = stratified_split(rows, (VAL_RATIO, TEST_RATIO), args.seed)

    all_rows = train + val + test
    counts = Counter(r["label"] for r in all_rows)
    print("\n전체 등급 분포")
    for lv in sorted(LABEL_NAMES):
        n = counts.get(lv, 0)
        print(f"  {lv} {LABEL_NAMES[lv]:<10} {n:>6}건" + ("   ⚠ 부족" if n < 20 else ""))
    if any(counts.get(lv, 0) == 0 for lv in LABEL_NAMES):
        print("\n  ⚠ 데이터가 0건인 등급이 있습니다. 모델은 그 등급을 절대 예측하지 못합니다.")
    if not test:
        print("\n  ⚠ test 가 비었습니다 — 사람이 매긴 라벨이 한 건도 없습니다.")
        print("    이 상태로 학습하면 성능 수치를 만들 수 없습니다. `python3 label.py` 를 먼저 하세요.")
    elif len(test) < 50:
        print(f"\n  ⚠ test 가 {len(test)}건뿐입니다. 100건 아래면 정확도 오차가 ±10%p 안팎으로 흔들립니다.")

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

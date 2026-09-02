"""사람 라벨과 LLM 자동 라벨이 얼마나 일치하는지 잰다.

자동 라벨을 학습에 쓰기 전에 반드시 이걸로 확인한다.
일치율이 우연 수준(3단계면 33%)이면 그 라벨로 학습해봐야 LLM의 편향만 배운다.

프롬프트를 고칠 때는 dev 로만 보고 고치고, 마지막에 holdout 으로 확인한다.
같은 데이터를 보고 고치면서 그 데이터로 성능을 재면 좋아 보이기만 할 뿐이다.

    python3 eval_autolabel.py                    # 전체
    python3 eval_autolabel.py --split dev        # 앞쪽 절반 (프롬프트 튜닝용)
    python3 eval_autolabel.py --split holdout    # 뒤쪽 절반 (최종 확인용)
    python3 eval_autolabel.py --auto data/labeled/try2.jsonl --split dev
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from config import AUTO_LABEL_PATH, LABEL_NAMES, LABELED_PATH, SEED


def load_jsonl(path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_ids(ids: list[int], split: str, seed: int) -> list[int]:
    """id를 고정 시드로 섞어 절반씩 나눈다. 실행할 때마다 같은 분할이 나온다."""
    if split == "all":
        return ids
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    chosen = shuffled[:half] if split == "dev" else shuffled[half:]
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 라벨 vs LLM 라벨 비교")
    parser.add_argument("--auto", default=str(AUTO_LABEL_PATH), help="비교할 자동 라벨 파일")
    parser.add_argument("--human", default=str(LABELED_PATH))
    parser.add_argument("--split", choices=["all", "dev", "holdout"], default="all")
    parser.add_argument("--show", type=int, default=10, help="불일치 예시 개수")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    human_rows = load_jsonl(args.human)
    human = {int(r["id"]): int(r["label"]) for r in human_rows}
    auto = {int(r["id"]): int(r["label"]) for r in load_jsonl(args.auto)}
    titles = {int(r["id"]): r.get("title", "") for r in human_rows}

    common = sorted(set(human) & set(auto))
    if not common:
        raise SystemExit("겹치는 기사가 없습니다. autolabel.py --only-labeled 를 먼저 실행하세요.")

    ids = split_ids(common, args.split, args.seed)
    levels = sorted(LABEL_NAMES)

    exact = sum(1 for i in ids if human[i] == auto[i])
    close = sum(1 for i in ids if abs(human[i] - auto[i]) <= 1)
    chance = 1 / len(levels)

    print(f"[{args.split}] 비교 대상 {len(ids)}건  (파일: {args.auto})")
    print(f"  정확 일치    {exact}/{len(ids)} = {exact / len(ids):.1%}   (우연 수준 {chance:.0%})")
    if len(levels) > 3:
        print(f"  ±1등급 이내  {close / len(ids):.1%}")

    print("\n혼동 행렬 (행=사람, 열=LLM)")
    print("            " + "".join(f"LLM{c:>5}" for c in levels))
    for h in levels:
        row = [sum(1 for i in ids if human[i] == h and auto[i] == a) for a in levels]
        print(f"  사람 {h} {LABEL_NAMES[h]:<6}" + "".join(f"{v:>8}" for v in row) + f"   (계 {sum(row)})")

    print("\n분포 비교")
    hc, ac = Counter(human[i] for i in ids), Counter(auto[i] for i in ids)
    for lv in levels:
        h_n, a_n = hc.get(lv, 0), ac.get(lv, 0)
        gap = a_n - h_n
        flag = "  ← LLM 과다" if gap > len(ids) * 0.1 else ("  ← LLM 과소" if gap < -len(ids) * 0.1 else "")
        print(f"  {lv} {LABEL_NAMES[lv]:<6} 사람 {h_n:>3}건  vs  LLM {a_n:>3}건{flag}")

    # 어느 방향으로 틀리는지 — 프롬프트를 어떻게 고칠지 알려주는 신호
    over = sum(1 for i in ids if auto[i] > human[i])
    under = sum(1 for i in ids if auto[i] < human[i])
    print(f"\n오차 방향: 과대평가 {over}건 / 과소평가 {under}건")

    mism = [i for i in ids if human[i] != auto[i]]
    if mism and args.show:
        print(f"\n불일치 예시 (전체 {len(mism)}건 중 {min(args.show, len(mism))}건)")
        for i in random.Random(args.seed).sample(mism, min(args.show, len(mism))):
            print(f"  사람 {human[i]} / LLM {auto[i]}  |  {titles.get(i, '')[:56]}")

    if exact / len(ids) < chance + 0.15:
        print("\n  ⚠ 우연 수준과 큰 차이가 없습니다. 이 자동 라벨로 학습하면 안 됩니다.")


if __name__ == "__main__":
    main()

"""사람 라벨과 LLM 라벨이 얼마나 일치하는지 잰다.

기준이 되는 쪽은 **origin="human" 인 라벨뿐**이다. LLM 라벨을 기준으로 LLM 을 재면
"이 모델이 저 모델을 얼마나 흉내내나"가 나올 뿐, 맞는지 틀린지는 영영 알 수 없다.

자동 라벨을 학습에 쓰기 전에 반드시 이걸로 확인한다.
일치율이 우연 수준(3단계면 33%)이면 그 라벨로 학습해봐야 LLM의 편향만 배운다.

프롬프트를 고칠 때는 dev 로만 보고 고치고, 마지막에 holdout 으로 확인한다.
같은 데이터를 보고 고치면서 그 데이터로 성능을 재면 좋아 보이기만 할 뿐이다.

    python3 eval_autolabel.py                    # 전체
    python3 eval_autolabel.py --split dev        # 앞쪽 절반 (프롬프트 튜닝용)
    python3 eval_autolabel.py --split holdout    # 뒤쪽 절반 (최종 확인용)
    python3 eval_autolabel.py --auto data/labeled/try2.jsonl --split dev
    python3 eval_autolabel.py --auto data/labeled/backend_labels.jsonl   # 백엔드 시드 품질
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from autolabel import EXAMPLE_IDS
from config import AUTO_LABEL_PATH, HUMAN_ORIGIN, LABEL_NAMES, LABELED_PATH, SEED


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

    all_rows = load_jsonl(args.human)
    human_rows = [r for r in all_rows if r.get("origin") == HUMAN_ORIGIN]
    dropped = len(all_rows) - len(human_rows)
    if dropped:
        origins = Counter(r.get("origin", "unknown") for r in all_rows if r.get("origin") != HUMAN_ORIGIN)
        print(f"⚠ 사람이 매기지 않은 {dropped}건을 기준에서 제외했습니다 "
              f"({', '.join(f'{k} {v}건' for k, v in origins.most_common())}).")
    if not human_rows:
        raise SystemExit(
            f"사람이 매긴 라벨이 없습니다. ({args.human})\n"
            "`python3 label.py` 로 정답지를 먼저 만드세요. "
            "LLM 라벨을 기준으로 재면 나오는 숫자는 정확도가 아닙니다."
        )
    human = {int(r["id"]): int(r["label"]) for r in human_rows}
    auto = {int(r["id"]): int(r["label"]) for r in load_jsonl(args.auto)}
    titles = {int(r["id"]): r.get("title", "") for r in human_rows}

    common = sorted(set(human) & set(auto))
    if not common:
        raise SystemExit("겹치는 기사가 없습니다. autolabel.py --only-labeled 를 먼저 실행하세요.")

    # 프롬프트에 정답을 보여준 기사는 채점에서 뺀다. 안 그러면 실력이 아니라 암기를 잰다.
    leaked = [i for i in common if i in EXAMPLE_IDS]
    if leaked:
        print(f"⚠ 프롬프트 few-shot 예시 {len(leaked)}건이 정답지에 섞여 있어 채점에서 제외합니다.")
        print(f"  {leaked}")
        print("  이 기사들은 LLM 에게 답을 미리 보여준 것이라 맞히는 게 당연합니다.")
        common = [i for i in common if i not in EXAMPLE_IDS]
        if not common:
            raise SystemExit("예시를 빼고 나니 비교할 기사가 없습니다.")

    ids = split_ids(common, args.split, args.seed)
    levels = sorted(LABEL_NAMES)

    exact = sum(1 for i in ids if human[i] == auto[i])
    close = sum(1 for i in ids if abs(human[i] - auto[i]) <= 1)
    # 불균형할수록 "다 0으로 찍기"가 강한 기준선이 된다. 우연(1/등급수)보다 이쪽이 정직하다.
    hc = Counter(human[i] for i in ids)
    majority = max(hc.values()) / len(ids)

    print(f"[{args.split}] 비교 대상 {len(ids)}건  (파일: {args.auto})")
    print(f"  정확 일치    {exact}/{len(ids)} = {exact / len(ids):.1%}")
    print(f"  기준선       {majority:.1%}  (전부 '{LABEL_NAMES[max(hc, key=lambda k: hc[k])]}'로 찍었을 때)")
    if exact / len(ids) <= majority:
        print("  ⚠ 다 찍기보다 못합니다. 이 자동 라벨은 정보를 전혀 더하지 않습니다.")
    if len(levels) > 3:
        print(f"  ±1등급 이내  {close / len(ids):.1%}")

    if len(levels) == 2:
        # 인기기사 자리에 올릴지 말지가 목적이므로, 1(중요)에 대한 정밀도·재현율이 본질이다.
        tp = sum(1 for i in ids if human[i] == 1 and auto[i] == 1)
        fp = sum(1 for i in ids if human[i] == 0 and auto[i] == 1)
        fn = sum(1 for i in ids if human[i] == 1 and auto[i] == 0)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"\n'1 중요' 기준 (인기기사 자리에 올릴지의 판단)")
        print(f"  정밀도 {prec:.1%}  — 1로 고른 것 중 실제로 중요한 비율 (올렸을 때 안 민망한가)")
        print(f"  재현율 {rec:.1%}  — 실제 중요 기사 중 잡아낸 비율 (놓친 게 얼마나 되나)")
        print(f"  F1     {f1:.1%}   (TP {tp} / FP {fp} / FN {fn})")

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

    if exact / len(ids) < majority + 0.05:
        print("\n  ⚠ 다 찍기 기준선과 큰 차이가 없습니다. 이 자동 라벨로 학습하면 안 됩니다.")


if __name__ == "__main__":
    main()

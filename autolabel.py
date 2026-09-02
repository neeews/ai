"""Ollama(exaone3.5)로 기사에 중요도 0~4를 자동으로 매긴다 — 학습 데이터 대량 확보용.

여기서 나온 라벨은 어디까지나 **학습용 근사치**다. 성능 측정은 반드시 사람이 매긴
data/labeled/labels.jsonl 로 해야 한다. (안 그러면 "LLM 흉내를 얼마나 잘 내나"만 재게 된다)

    python3 autolabel.py                  # 라벨 없는 기사 전부
    python3 autolabel.py --limit 500
    python3 autolabel.py --model qwen3:4b
    nohup python3 autolabel.py > autolabel.log 2>&1 &   # 오래 걸리므로 백그라운드 권장

한 건 처리할 때마다 즉시 저장하므로 중간에 끊겨도 이어서 실행하면 된다.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from config import (
    AUTO_LABEL_PATH,
    LABEL_NAMES,
    LABEL_SCHEME,
    NUM_LABELS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RAW_PATH,
)

PROMPT_3 = """당신은 뉴스 편집장이다. 아래 기사의 중요도를 0, 1, 2 중 하나로 평가하라.

[등급 기준]
2 = 중요: 다수에게 실질적 영향. 주요 정책·법안, 대형 사건사고, 재난, 국가 단위 정치·경제 변화
1 = 보통: 특정 산업·집단에 의미 있음. 업계 판도를 바꾸는 기업 소식, 주요 지역 행정, 굵직한 국제 동향
0 = 낮음: 소수 관심사이거나 정보 가치가 없음. 개별 기업의 일상 소식, 홍보성 기사,
    기부·시상, 지점 개설, 연예인 근황, 인사 발령, 단순 시황, 부고, 날씨 단신

[매우 중요한 규칙]
목표 분포는 0등급 약 35%, 1등급 약 40%, 2등급 약 25% 다. 2를 아껴서 써라.
"전 국민 다수가 오늘 알아야 하는가?"에 아니오면 2를 주지 마라.
두 등급 사이에서 망설여지면 반드시 낮은 쪽을 골라라.
제목이 거창해도 내용이 한 회사의 홍보면 0이다.

[예시]
"스마일게이트 로드나인, 지역 어르신 지원 기부금 전달" -> 0
"메리츠증권, 부산금융센터 해운대로 이전 오픈" -> 0
"하나은행, 미국으로 해외 송금 단 1분만에" -> 0
"국토부 대광위원장, 전주 기린대로 간선급행버스체계 현장 점검" -> 0
"모더나 mRNA 항암 백신에 열광…암 정복까진 갈 길 멀다" -> 1
"백악관, 빅테크와 AI 안전성 시험 체계 논의" -> 1
"푸틴 러일 관계 악화는 일본 책임…日 우크라이나 침공 탓" -> 1
"법원, 홈플러스 회생계획안 인가…공익채권자 75.9% 동의" -> 2
"부산시, 예인선 전복 사고수습본부 가동…구조에 총력" -> 2
"정부, 종합부동산세 개편안 발표…1주택자 세부담 30% 경감" -> 2

[기사]
카테고리: {category}
제목: {title}
본문: {body}

숫자 하나만 출력하라. 설명, 문장, 기호를 붙이지 마라.
답:"""

PROMPT_5 = """당신은 뉴스 편집장이다. 아래 기사의 중요도를 0~4 중 하나로 평가하라.

[등급 기준]
4 = 매우 중요: 전 국민·국제적 영향. 대규모 재난, 전쟁, 금리 급변, 국가 단위 정치 격변
3 = 높음: 다수에게 실질적 영향. 주요 정책 발표, 대기업 구조조정, 법안 통과, 대형 사건사고
2 = 보통: 특정 산업·집단에 의미 있음. 업계 판도를 바꾸는 기업 소식, 주요 지역 행정
1 = 낮음: 소수 관심사. 개별 기업의 일상적 소식, 연예인 근황, 소규모 행사, 인사 발령
0 = 무시해도 됨: 정보 가치 없음. 홍보성 기사, 기부·시상 소식, 지점 개설, 부고, 날씨 단신

[매우 중요한 규칙]
실제 뉴스의 대부분은 0~1이다. 다음 분포를 지켜라.
  0등급 약 25%, 1등급 약 40%, 2등급 약 25%, 3등급 약 8%, 4등급 약 2%
3과 4는 아껴서 써라. "전 국민이 오늘 당장 알아야 하는가?"에 아니오면 3 이상을 주지 마라.
두 등급 사이에서 망설여지면 반드시 낮은 쪽을 골라라.
제목이 거창해도 내용이 한 회사의 홍보면 0이다.

[예시]
"스마일게이트 로드나인, 지역 어르신 지원 기부금 전달" -> 0
"메리츠증권, 부산금융센터 해운대로 이전 오픈" -> 0
"하나은행, 미국으로 해외 송금 단 1분만에" -> 1
"햇반이 아이스크림으로…CJ제일제당 IP 제품 700만개 팔았다" -> 1
"모더나 mRNA 항암 백신에 열광…암 정복까진 갈 길 멀다" -> 2
"백악관, 빅테크와 AI 안전성 시험 체계 논의" -> 2
"정부, 종합부동산세 개편안 발표…1주택자 세부담 30% 경감" -> 3
"경북 산불 사흘째 확산, 사망 26명·이재민 3만명" -> 4

[기사]
카테고리: {category}
제목: {title}
본문: {body}

숫자 하나만 출력하라. 설명, 문장, 기호를 붙이지 마라.
답:"""

# config.LABEL_SCHEME 에 맞는 프롬프트를 고른다
PROMPT = PROMPT_3 if LABEL_SCHEME == "3" else PROMPT_5


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ask_ollama(url: str, model: str, prompt: str, timeout: int) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        # 숫자 하나만 받으면 되므로 출력 길이를 최소로 → CPU에서 크게 빨라진다
        "options": {"temperature": 0.0, "num_predict": 8, "top_p": 1.0},
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("response", "")


def parse_label(text: str) -> int | None:
    """모델이 '답: 3' 이나 '3점' 처럼 답해도 첫 숫자를 뽑아낸다."""
    m = re.search(rf"[0-{NUM_LABELS - 1}]", text)
    return int(m.group()) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 자동 라벨링")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--limit", type=int, help="이번 실행에서 처리할 최대 건수")
    parser.add_argument("--body-chars", type=int, default=1200,
                        help="본문을 몇 자까지 넣을지. 짧을수록 빠르다 (기본 1200)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", help="저장 경로 (기본: data/labeled/auto_labels.jsonl)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else AUTO_LABEL_PATH

    articles = load_jsonl(RAW_PATH)
    if not articles:
        print(f"기사 풀이 비어 있습니다. `python3 export_db.py` 를 먼저 실행하세요. ({RAW_PATH})")
        return

    done_ids = {int(r["id"]) for r in load_jsonl(out_path)}
    todo = [a for a in articles if int(a["id"]) not in done_ids]
    if args.limit:
        todo = todo[:args.limit]

    if not todo:
        print("자동 라벨링할 기사가 없습니다.")
        return

    print(f"모델 {args.model} | 대상 {len(todo)}건 (완료 {len(done_ids)}건)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[int] = Counter()
    failed = 0
    started = time.time()

    with out_path.open("a", encoding="utf-8") as out:
        for i, article in enumerate(todo, start=1):
            prompt = PROMPT.format(
                category=article.get("category", ""),
                title=article["title"],
                body=(article.get("body") or "")[:args.body_chars],
            )
            try:
                raw = ask_ollama(args.url, args.model, prompt, args.timeout)
                label = parse_label(raw)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"  [{i}] 요청 실패: {exc}")
                failed += 1
                continue

            if label is None:
                print(f"  [{i}] 숫자 파싱 실패: {raw[:40]!r}")
                failed += 1
                continue

            out.write(json.dumps({
                "id": int(article["id"]),
                "label": label,
                "title": article["title"],
                "category": article.get("category", ""),
                "model": args.model,
            }, ensure_ascii=False) + "\n")
            out.flush()
            counts[label] += 1

            if i % 20 == 0 or i == len(todo):
                elapsed = time.time() - started
                speed = i / elapsed
                remain = (len(todo) - i) / speed if speed else 0
                dist = " ".join(f"{lv}:{counts.get(lv, 0)}" for lv in sorted(LABEL_NAMES))
                print(f"  {i}/{len(todo)}  {speed:.2f}건/초  남은시간 {remain / 60:.0f}분  [{dist}]",
                      flush=True)

    print(f"\n완료: 성공 {sum(counts.values())}건, 실패 {failed}건 → {out_path}")
    for lv in sorted(LABEL_NAMES):
        print(f"  {lv} {LABEL_NAMES[lv]:<10} {counts.get(lv, 0):>5}건")


if __name__ == "__main__":
    main()

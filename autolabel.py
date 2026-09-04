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
    LABELED_PATH,
    LABEL_SCHEME,
    NUM_LABELS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RAW_PATH,
)

# 아래 프롬프트의 few-shot 예시로 쓴 기사들. LLM 에게 정답을 미리 보여준 기사이므로
# 기사 풀에도, 정답지에도 절대 들어가면 안 된다 — 들어가면 일치율이 부풀려진다.
# (실제로 이전 3단계 프롬프트에서 예시 9건이 정답지에 섞여 그 9건만 78~89%,
#  나머지 91건은 45~48% 로 30%p 넘게 벌어졌다.)
# 예시를 바꾸면 이 목록도 같이 바꿔야 한다.
EXAMPLE_IDS: frozenset[int] = frozenset({
    60959, 25394, 58272, 10254, 64969, 46413, 2831, 62903, 65861, 37808, 24681,  # -> 0
    55297, 37184, 65245, 31823,                                                   # -> 1
})

PROMPT_2 = """당신은 뉴스 앱 메인 화면의 편집장이다. 아래 기사를 메인의 '오늘의 중요 뉴스'
자리에 올릴지 말지 정하라. 올릴 만하면 1, 아니면 0이다.

기준은 단 하나: **이 기사를 오늘 놓치면 독자가 손해를 보는가.**
슬프거나 자극적인 것, 클릭이 많이 나올 것과 중요한 것은 완전히 다르다.

[1 — 아래에 해당할 때만 1이다]
   - 전국 단위 정치 일정·격돌 (인사청문회, 국회 본회의, 대선·총선)
   - 장관·국회의원·검찰 고위직 등 유력 인사의 비리·의혹·기소
   - 사망·실종자가 여러 명인 대형 참사
   - 국민 다수가 아는 대기업의 존폐 (회생, 파산, 대규모 구조조정)
   - 사법·수사 제도 자체를 바꾸는 사건 (검찰 개편, 대법원 판례 변경)
   - 개인의 돈·자격·의무를 바꾸는 제도 변경 (세금, 보험료, 지원금, 면허 기준)

[0 — 위에 없으면 전부 0이다]
   - 특정 시·군·구 단위 소식, 개인 한 명의 사고, 특정 학교·기관 내부의 일
   - 부고·인사·수상·기부·봉사, 연예·스포츠 인터뷰와 신작 소개, 한 기업의 홍보
   - 금리·환율·시황, 해외 정상외교, 해외 사건사고
   - 개별 형사사건의 판결·송치, 기업 간 분쟁, 제품 회수
   → 사람이 죽은 사고라도 지역의 개별 사고면 0이다. 냉정하게 판단하라.

[반드시 기억할 것]
당신은 중요도를 **과대평가하는 경향**이 있다. 실제로 1은 100건 중 13건뿐이다.
망설여지면 무조건 0이다.

[예시]
"[부고] 남궁철(파이로 크리에이션즈 대표)씨 부친상" -> 0
"[인사] 한국산업은행" -> 0
"[부산소식] 부산시 아름다운 조경상 공모" -> 0
"[프로야구 수원전적] 롯데 4-0 kt" -> 0
"카카오게임즈, '도깨비의세계' 10월 출시…사전 등록 시작" -> 0
"울주군 청년창업기업 5곳, 수도권 혁신 창업 생태계 탐방" -> 0
"서울 노원구 화랑대사거리서 덤프트럭끼리 추돌…“구리방향 교통 통제”" -> 0
"김민석 \"협력을 위한 견제\"…전남광주시 집행부-의회 협치 주문" -> 0
"[마켓뷰] 또 오른 美국채 금리…전날 급등했던 코스피 발목 잡나" -> 0
"무기 재고, 공습 효과, 전쟁 출구…다 막힌 트럼프" -> 0
"日 최초 노벨 생리의학상 도네가와 스스무 별세…면역 구조 규명" -> 0
"[속보] 합참 “북한, 동쪽 방향으로 미상 발사체 발사”" -> 1
"[속보] 법원 \"尹, 윤우진에게 변호사 소개했다고 봐야\"" -> 1
"국가교육위 \"수능 서논술형 도입 확정 아냐…국민 의견수렴 중\"" -> 1
"정부 \"제주항공 참사 재수색 일정차질 불가피하나 신속 마무리\"" -> 1

[평가할 기사]
카테고리: {category}
제목: {title}
본문: {body}

0 또는 1 한 글자만 출력하라. 설명, 문장, 기호를 붙이지 마라.
답:"""


PROMPT_3 = """당신은 전국 종합일간지의 1면 편집장이다. 아래 기사의 중요도를 0, 1, 2 중 하나로 매겨라.

기준은 단 하나: **이 기사가 전국 독자에게 얼마나 중요한가.**
기사가 슬프거나 자극적인 것과 중요한 것은 완전히 다르다.

[판정 절차 — 위에서부터 순서대로 적용하고, 걸리면 즉시 확정한다]

1단계. 다음에 하나라도 해당하면 무조건 **0**이다.
   - 특정 시·군·구·읍·면 단위의 소식 (지역 의회 건의안, 조례 발의, 지역 행사, 지역 민원)
   - 개인 한 명의 사고·부상·사망·입건 (교통사고, 추락, 실족)
   - 특정 학교·대학·기관 내부의 일 (실험실 화재, 역명 변경 요청, 돌봄 프로그램)
   - 부고, 인사, 수상·포장, 기부·나눔, 봉사
   - 연예·문화·스포츠의 인터뷰, 신작 소개, 감독·선수 코멘트
   - 특정 기업 한 곳의 홍보성 소식 (신제품, 지점 개설, 수상)
   → 사람이 죽은 사고라도 지역의 개별 사고면 0이다. 냉정하게 판단하라.

2단계. 다음에 해당하면 **2**다. 여기 해당하지 않으면 절대 2를 주지 마라.
   - 전국 단위 정치 일정·격돌 (인사청문회, 국회 본회의, 대선·총선)
   - 장관·국회의원·검찰 고위직 등 유력 인사의 비리·의혹·기소
   - 사망·실종자가 여러 명인 대형 참사
   - 국민 다수가 아는 대기업의 존폐 (회생, 파산, 대규모 구조조정)
   - 사법·수사 제도 자체를 바꾸는 사건 (검찰 개편, 대법원 판례 변경)

3단계. 위 둘 다 아니면 **1**이다.
   - 산업·경제 동향, 금리·환율·시황, 기업 간 분쟁
   - 해외 뉴스 (국제 정상외교, 해외 사건사고, 해외 재판)
   - 개별 형사사건의 재판 결과, 검찰 송치
   - 특정 산업·집단이 입은 피해

[반드시 기억할 것]
당신은 중요도를 **과대평가하는 경향**이 있다. 실제 분포는 0이 35%, 1이 40%, 2가 25%다.
망설여지면 무조건 낮은 쪽을 골라라. 특히 2를 줄지 1을 줄지 망설이면 1이다.

[예시]
"의령군의회 농어촌 기본소득 단계적 확대·법제화해야 건의안" -> 0
"경기 광주서 농지 조사하던 50대 트럭에 치여 사망…운전자 입건" -> 0
"성균관대학교 자연과학캠퍼스 실험실서 불…인명피해 없어" -> 0
"[충북소식] 오경숙 도 양성평등가족정책관 국민포장" -> 0
"구미 여름방학 틈새돌봄 8천500명 이용…겨울도 운영" -> 0
"김원형 두산 감독, 사구에도 1루로 향한 김민석에 근성 보여줘" -> 0
"유가 급등·대외금리 상승에 국고채 금리↑…3년물 연 3.930%" -> 1
"시진핑, 10년 만의 이집트 국빈방문…미·중 정상회담 앞두고" -> 1
"홍콩 민주화운동가 조슈아 웡, 외국과 공모 혐의 유죄 인정" -> 1
"박대준 쿠팡 전 대표 국회 위증 혐의 검찰 송치" -> 1
"식약처, 잔류농약 기준 초과 중국산 목이버섯 회수" -> 1
"추석 연휴 한주 전 인사청문 슈퍼위크…여야 격돌 예고" -> 2
"부산 앞바다 예인선 전복 사고…1명 사망·6명 실종" -> 2
"법원, 홈플러스 회생계획안 인가…채권자 3분의 2 이상 동의" -> 2
"검찰, 박성주 전 국수본부장 기소…장윤기 부실수사 지휘 혐의" -> 2

[평가할 기사]
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

# --- 이분 질문 방식 ---
# 작은 모델은 3지선다 채점표를 따라가지 못하고 전부 "중요"로 몰아버린다.
# 대신 예/아니오 질문 두 번으로 쪼개면 각 판단이 단순해져 훨씬 잘 따라온다.

Q_LOCAL = """다음 뉴스가 특정 지역(시·군·구), 특정 개인 한 명, 또는 특정 기관 내부에만
관련된 소식이면 Y, 전국 독자에게 의미가 있으면 N 이라고 답하라.

Y 인 것들: 지역 의회 건의안·조례, 지역 행사·민원, 개인의 교통사고·추락사고,
특정 학교의 화재, 부고, 인사·수상, 기부·봉사, 연예인·감독 인터뷰, 한 기업의 홍보

N 인 것들: 전국 정치, 국가 경제·금리, 해외 뉴스, 산업 전반 동향, 대기업의 존폐, 대형 참사

제목: {title}
본문: {body}

Y 또는 N 한 글자만 출력하라."""

Q_FRONTPAGE = """다음 뉴스가 오늘 전국 종합일간지 **1면 머리기사**가 될 만하면 Y,
안쪽 지면에 실릴 정도면 N 이라고 답하라.

Y 인 것들: 국회 인사청문·본회의 격돌, 장관·검찰 고위직의 비리·기소,
사망·실종자 여러 명인 대형 참사, 국민 다수가 아는 대기업의 회생·파산, 사법제도 개편

N 인 것들: 금리·환율·시황, 해외 정상외교, 해외 사건사고, 개별 형사사건 판결·송치,
특정 산업의 피해, 기업 간 분쟁, 제품 회수

1면 머리기사는 하루에 한두 건뿐이다. 애매하면 N 이다.

제목: {title}
본문: {body}

Y 또는 N 한 글자만 출력하라."""

# config.LABEL_SCHEME 에 맞는 프롬프트를 고른다
PROMPT = {"2": PROMPT_2, "3": PROMPT_3, "5": PROMPT_5}[LABEL_SCHEME]


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


def parse_yes(text: str) -> bool | None:
    """Y/N 응답을 읽는다. 모델이 '예'/'아니오'로 답해도 받아준다."""
    t = text.strip().upper()
    if not t:
        return None
    if t.startswith(("Y", "예")) or "YES" in t:
        return True
    if t.startswith(("N", "아니")) or "NO" in t:
        return False
    return None


def label_by_cascade(url: str, model: str, article: dict, body_chars: int, timeout: int) -> int | None:
    """예/아니오 질문만으로 등급을 정한다. 작은 모델은 다지선다보다 이분 판단에 강하다.

    2단계: "1면 머리기사감인가?" 한 번이면 끝난다. 지역·개인 소식을 걸러내는 질문은
    어차피 1면감이 아니라 같은 답이 나오므로 생략한다 — 건당 LLM 호출이 절반으로 준다.
    3·5단계: 지역·개인 소식이면 최하위, 1면감이면 최상위, 나머지는 가운데.
    """
    fields = {"title": article["title"], "body": (article.get("body") or "")[:body_chars]}
    top = NUM_LABELS - 1
    mid = 0 if NUM_LABELS == 2 else 1

    if NUM_LABELS > 2:
        is_local = parse_yes(ask_ollama(url, model, Q_LOCAL.format(**fields), timeout))
        if is_local is None:
            return None
        if is_local:
            return 0

    is_front = parse_yes(ask_ollama(url, model, Q_FRONTPAGE.format(**fields), timeout))
    if is_front is None:
        return None
    return top if is_front else mid


def parse_label(text: str) -> int | None:
    """모델이 '답: 3' 이나 '3점' 처럼 답해도 첫 숫자를 뽑아낸다."""
    m = re.search(rf"[0-{NUM_LABELS - 1}]", text)
    return int(m.group()) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 자동 라벨링")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--limit", type=int, help="이번 실행에서 처리할 최대 건수")
    parser.add_argument("--only-labeled", action="store_true",
                        help="사람이 라벨한 기사에만 실행 — 프롬프트 정확도 측정용")
    parser.add_argument("--body-chars", type=int, default=1200,
                        help="본문을 몇 자까지 넣을지. 짧을수록 빠르다 (기본 1200)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--strategy", choices=["single", "cascade"], default="single",
                        help="single=한 번에 등급 물음, cascade=예/아니오 두 번 (3단계 전용)")
    parser.add_argument("--out", help="저장 경로 (기본: data/labeled/auto_labels.jsonl)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else AUTO_LABEL_PATH

    articles = load_jsonl(RAW_PATH)
    if not articles:
        print(f"기사 풀이 비어 있습니다. `python3 export_db.py` 를 먼저 실행하세요. ({RAW_PATH})")
        return

    done_ids = {int(r["id"]) for r in load_jsonl(out_path)}
    todo = [a for a in articles if int(a["id"]) not in done_ids]
    if args.only_labeled:
        human_ids = {int(r["id"]) for r in load_jsonl(LABELED_PATH)}
        if not human_ids:
            print(f"사람 라벨이 없습니다. ({LABELED_PATH})")
            return
        todo = [a for a in todo if int(a["id"]) in human_ids]
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
            prompt = "" if args.strategy == "cascade" else PROMPT.format(
                category=article.get("category", ""),
                title=article["title"],
                body=(article.get("body") or "")[:args.body_chars],
            )
            try:
                if args.strategy == "cascade":
                    label = label_by_cascade(
                        args.url, args.model, article, args.body_chars, args.timeout
                    )
                    raw = ""
                else:
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

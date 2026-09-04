"""프로젝트 전역 설정 — 경로, 모델, 하이퍼파라미터, 라벨 정의."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "articles.jsonl"            # DB에서 뽑아온 기사 풀
AUTO_LABEL_PATH = DATA_DIR / "labeled" / "auto_labels.jsonl"   # 로컬 LLM(Ollama)이 매긴 라벨 (학습용)
BACKEND_LABEL_PATH = DATA_DIR / "labeled" / "backend_labels.jsonl"  # 백엔드에서 반입한 라벨 (학습용)
LABELED_PATH = DATA_DIR / "labeled" / "labels.jsonl"      # 사람이 매긴 라벨 (평가 정답지 — 사람 것만)

# 정답지(test)에 들어갈 수 있는 유일한 출처. 모든 라벨 행은 origin 을 반드시 들고 다닌다.
HUMAN_ORIGIN = "human"
# 백엔드 시드(article_importance_labels)를 실제로 매긴 주체. DB에 기록이 없어 여기에 적어 둔다.
# 2026-09-02 c616972 커밋 기준 claude-opus-5 가 매겼고, 사람 검수는 거치지 않았다.
BACKEND_LABELED_BY = "claude-opus-5"
SPLIT_DIR = DATA_DIR / "splits"

# 학습시킨 모델의 이름. 폴더명·로그·서빙에서 이 이름 하나로 부른다.
# 여러 판본을 비교할 거면 MODEL_NAME 에 판본을 달아라 (예: "muni-v2").
# 그냥 두면 다시 학습할 때 같은 폴더를 덮어써서 이전 모델이 사라진다.
MODEL_NAME = "muni"
MODEL_DIR = ROOT / "models" / MODEL_NAME

# KoELECTRA v3 discriminator. GPU 없는 VPS라면 --model 로 small 을 지정해 속도를 5배쯤 올릴 수 있다.
BASE_MODEL = "monologg/koelectra-base-v3-discriminator"
SMALL_MODEL = "monologg/koelectra-small-v3-discriminator"

# 등급 체계. "2" / "3" / "5" 중 하나로 바꾸면 라벨링·학습·평가가 통째로 따라 바뀐다.
#
# 기본값이 "2"인 이유: 이 모델의 쓰임은 메인의 인기기사 자리에 올릴 기사를 고르는 것
# 하나뿐이고, 그 자리는 결국 "올린다/안 올린다"의 이진 결정이다. 등급을 잘게 나눌수록
# 경계(특히 3단계의 MEDIUM)에서 사람도 LLM도 흔들려 라벨 품질만 떨어진다.
# 순위가 필요하면 등급이 아니라 predict.py 의 확률값으로 정렬한다.
LABEL_SCHEME = "2"

_SCHEMES: dict[str, dict[int, str]] = {
    "2": {
        0: "일반",
        1: "중요",
    },
    "3": {
        0: "낮음",
        1: "보통",
        2: "중요",
    },
    "5": {
        0: "무시해도 됨",
        1: "낮음",
        2: "보통",
        3: "높음",
        4: "매우 중요",
    },
}

LABEL_NAMES = _SCHEMES[LABEL_SCHEME]
NUM_LABELS = len(LABEL_NAMES)

# 백엔드 enum 이름 ↔ 학습용 정수 라벨.
# 2단계에서 MEDIUM 을 0 에 붙이는 이유: 인기기사 자리는 상위 소수만 노출하므로 문턱이 높아야 한다.
# 시드 300건 기준 HIGH 40건(13%)이 양성이 되는데, 이게 하루 노출 건수와 얼추 맞는다.
_BACKEND_TO_LABEL: dict[str, dict[str, int]] = {
    "2": {"LOW": 0, "MEDIUM": 0, "HIGH": 1},
    "3": {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    "5": {"LOW": 1, "MEDIUM": 2, "HIGH": 3},
}
_LABEL_TO_BACKEND: dict[str, dict[int, str]] = {
    "2": {0: "LOW", 1: "HIGH"},
    "3": {0: "LOW", 1: "MEDIUM", 2: "HIGH"},
    "5": {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "HIGH"},
}
BACKEND_TO_LABEL = _BACKEND_TO_LABEL[LABEL_SCHEME]
LABEL_TO_BACKEND = _LABEL_TO_BACKEND[LABEL_SCHEME]

# 인기기사 자리에 한 번에 노출하는 건수. precision@N 을 이 값으로 잰다.
SERVE_TOP_N = 10

# 하이퍼파라미터 (GPU 유무는 train.py 가 자동 감지해 batch/fp16 조정)
MAX_LENGTH = 256
LEARNING_RATE = 3e-5
EPOCHS = 5
BATCH_SIZE_GPU = 16
BATCH_SIZE_CPU = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

VAL_RATIO = 0.15
TEST_RATIO = 0.15

# LLM 자동 라벨링 (VPS의 Ollama)
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "exaone3.5:2.4b"

"""프로젝트 전역 설정 — 경로, 모델, 하이퍼파라미터, 라벨 정의."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "articles.jsonl"            # DB에서 뽑아온 기사 풀
AUTO_LABEL_PATH = DATA_DIR / "labeled" / "auto_labels.jsonl"   # LLM이 매긴 라벨 (학습용)
LABELED_PATH = DATA_DIR / "labeled" / "labels.jsonl"      # 사람이 매긴 라벨 (평가 기준)
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = ROOT / "models" / "koelectra-importance"

# KoELECTRA v3 discriminator. GPU 없는 VPS라면 --model 로 small 을 지정해 속도를 5배쯤 올릴 수 있다.
BASE_MODEL = "monologg/koelectra-base-v3-discriminator"
SMALL_MODEL = "monologg/koelectra-small-v3-discriminator"

# 등급 체계. 백엔드(Importance enum)는 3단계(LOW/MEDIUM/HIGH)를 쓰고 있다.
# 여기를 "3" 또는 "5" 로 바꾸면 라벨링·학습·평가가 통째로 따라 바뀐다.
LABEL_SCHEME = "3"

_SCHEMES: dict[str, dict[int, str]] = {
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

# 백엔드 enum 이름 ↔ 학습용 정수 라벨. 3단계일 땐 그대로, 5단계일 땐 가운데로 펼쳐 넣는다.
BACKEND_TO_LABEL = (
    {"LOW": 0, "MEDIUM": 1, "HIGH": 2} if LABEL_SCHEME == "3"
    else {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
)
LABEL_TO_BACKEND = (
    {0: "LOW", 1: "MEDIUM", 2: "HIGH"} if LABEL_SCHEME == "3"
    else {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "HIGH"}
)

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

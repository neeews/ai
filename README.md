# 뉴스 중요도 분류 (KoELECTRA 파인튜닝)

뉴스 기사를 읽고 중요도를 판정하는 모델. `neeews` DB의 기사 79,866건을 학습 데이터로 쓴다.

등급 체계는 `config.py` 의 `LABEL_SCHEME` 하나로 바뀐다.

| 값 | 등급 | 비고 |
|---|---|---|
| `"3"` (기본) | 0 낮음 / 1 보통 / 2 중요 | 백엔드 `Importance` enum(LOW/MEDIUM/HIGH)과 1:1 |
| `"5"` | 0 무시 / 1 낮음 / 2 보통 / 3 높음 / 4 매우중요 | 더 세분화된 랭킹이 필요할 때 |

바꾸면 라벨링 CLI·프롬프트·학습·평가가 전부 따라 바뀐다.

## 실행 환경

DB와 Ollama가 올라가 있는 학습 서버의 `~/ai` 에서 전부 돌린다.
같은 호스트라 별도 터널이 필요 없다.

| 항목 | 검증한 환경 |
|---|---|
| CPU / RAM | 4코어 Xeon 5218R / 7.8GB (**GPU 없음**) |
| Python | 3.10 (`venv`) |
| DB | MySQL 8.4 `neeews.articles` — 127.0.0.1:3306 |
| LLM | Ollama `exaone3.5:2.4b` — 127.0.0.1:11434 |

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

DB 접속 정보는 백엔드의 `~/backend/.env` 를 그대로 읽는다
(`DB_HOST` `DB_PORT` `DB_USER` `DB_PASSWORD` `DB_NAME`). 환경변수로 덮어쓸 수도 있다.

로컬에서 고친 코드를 서버로 올리려면 `.deploy.env.example` 을 `.deploy.env` 로 복사해
접속 정보를 채운 뒤 `./sync.sh` 를 실행한다 (데이터·모델·venv는 서버에만 둔다).

## 전체 흐름

```
import_labels.py  백엔드가 만든 기존 라벨 반입  data/labeled/labels.jsonl
export_db.py   DB → 라벨링할 기사 풀        data/raw/articles.jsonl
     ↓
autolabel.py   LLM이 0~4 자동 라벨 (학습용)  data/labeled/auto_labels.jsonl
label.py       사람이 0~4 라벨 (평가 정답지) data/labeled/labels.jsonl
     ↓
prepare_dataset.py  train/val/test 분할       data/splits/*.jsonl
     ↓
train.py       KoELECTRA 파인튜닝            models/koelectra-importance/
     ↓
predict.py     새 기사 중요도 예측
```

**핵심 원칙**: 학습 라벨은 LLM이 대량으로 만들고, **평가 라벨은 사람이 만든다.**
LLM 라벨로 평가하면 "KoELECTRA가 exaone을 얼마나 잘 흉내내나"만 측정되고,
LLM이 틀린 부분을 영영 발견하지 못한다.

## 0. 기존 라벨 반입

백엔드가 이미 매겨 둔 라벨(`article_importance_labels` 테이블 / `labels/importance_seed.csv`)이 있으면
먼저 가져온다. 라벨만 있고 본문이 없으므로 기사 id로 DB에서 본문을 채워 넣는다.

```bash
python import_labels.py --from-db          # DB 테이블에서
python import_labels.py --csv path/to/importance_seed.csv --dry-run
```

`LOW/MEDIUM/HIGH` → 정수 라벨 변환은 `config.py` 의 `BACKEND_TO_LABEL` 이 담당한다.

## 1. 기사 추출

```bash
python export_db.py --n 3000 --min-len 300
```

카테고리(정치/경제/사회/세계/IT·과학/스포츠/연예문화/종합) 8개에서 균등하게 뽑는다.
`--min-len` 미만의 짧은 기사는 제외 — 본문이 100자도 안 되면 사람도 중요도를 못 정한다.
여러 번 실행해도 이미 뽑은 기사는 건너뛴다.

## 2-a. LLM 자동 라벨링 (학습용)

```bash
nohup python autolabel.py > autolabel.log 2>&1 &
tail -f autolabel.log
```

CPU에서 **약 7~8초/건**. 3,000건이면 6시간 남짓이므로 백그라운드로 돌린다.
한 건마다 즉시 저장하므로 중간에 끊겨도 다시 실행하면 이어서 한다.

프롬프트는 `autolabel.py` 상단의 `PROMPT`에 있다. LLM은 중요도를 부풀리는 경향이 강해서
목표 분포(0등급 25% / 1등급 40% / 2등급 25% / 3등급 8% / 4등급 2%)와 실제 예시를
프롬프트에 박아 교정했다. 분포가 다시 한쪽으로 쏠리면 예시를 손보고
`--out` 으로 다른 파일에 뽑아 비교하면 된다.

## 2-b. 사람 라벨링 (평가용, 200건 권장)

```bash
python label.py --review    # LLM 라벨을 제안으로 띄우고 검수 — Enter면 인정, 숫자면 수정
python label.py             # 백지 상태로 직접 판단
python label.py --stats     # 분포 확인
```

판정 기준은 `LABELING_GUIDE.md`. 라벨링 중 `g` 를 누르면 그 자리에서 볼 수 있다.
기준이 흔들리면 모델도 흔들리므로, 애매하면 **낮은 등급**으로 통일한다.

## 3. 데이터셋 분할

```bash
python prepare_dataset.py
```

기본 hybrid 모드: train/val = LLM 라벨, test = 사람 라벨.
사람이 본 기사는 train/val에서 제외해 평가 오염을 막는다.
겹치는 기사가 있으면 **LLM-사람 일치율**을 함께 출력한다 — 자동 라벨을 믿어도 되는지의 근거다.

## 4. 학습

```bash
python train.py                                          # KoELECTRA-base
python train.py --model monologg/koelectra-small-v3-discriminator   # 5배 빠름
python train.py --epochs 8 --lr 2e-5 --max-length 128
```

- GPU 없으면 자동으로 CPU 설정(batch 8, fp16 off)으로 떨어진다
- 등급별 데이터 수가 불균형하므로 클래스 가중치를 기본 적용 (`--no-class-weights` 로 해제)
- val macro-F1 기준 early stopping, 최고 성능 체크포인트를 저장

**평가 지표**

| 지표 | 의미 |
|---|---|
| accuracy | 정확히 맞힌 비율 |
| macro_f1 | 등급별 F1의 평균. 적은 등급(3·4)을 무시하면 떨어진다 |
| **qwk** | 0~4가 순서를 가진 등급이므로 "1칸 차이"와 "4칸 차이"를 구분해 매기는 점수. 이 프로젝트의 주 지표 |
| adjacent_accuracy | ±1등급까지 맞다고 치는 관대한 정확도. 실사용 체감에 가깝다 |

## 5. 예측

```bash
python predict.py --db-latest 20          # 최근 기사 20건
python predict.py --id 81397
python predict.py --text "제목
본문..."
```

`score`는 확률 가중 평균(0~4 연속값)이라 기사 정렬·랭킹에 쓰기 좋다.
코드에서 쓸 때는 `predict.ImportanceScorer` 를 직접 import 하면 된다.

## 파일

| 파일 | 역할 |
|---|---|
| `config.py` | 경로·모델·하이퍼파라미터·라벨 정의 |
| `db.py` | MySQL 접속 (`~/backend/.env` 에서 인증정보를 읽어 저장소에 비밀번호를 두지 않음) |
| `export_db.py` | DB → 기사 풀 추출 |
| `autolabel.py` | Ollama 자동 라벨링 |
| `label.py` | 사람 라벨링 CLI |
| `prepare_dataset.py` | train/val/test 분할 |
| `train.py` | KoELECTRA 파인튜닝 |
| `predict.py` | 추론 |
| `import_labels.py` | 백엔드 라벨(LOW/MEDIUM/HIGH) 반입 |
| `LABELING_GUIDE.md` | 등급 판정 기준 |
| `sync.sh` | 로컬 → 서버 코드 동기화 |

## 알아둘 것

- KoELECTRA 체크포인트는 TF에서 변환돼 일부 가중치가 non-contiguous 로 올라온다.
  그대로 두면 저장 단계에서 죽으므로 `train.py` 가 로드 직후 `.contiguous()` 로 정렬한다.
- `requirements.txt` 의 버전은 서로 맞물려 있다. 특히 `accelerate` 가 없으면 `Trainer` 가 뜨지 않는다.
- LLM 자동 라벨은 중요도를 부풀리는 경향이 있다. 학습 전에 반드시 분포를 확인하고,
  0등급이 거의 안 나오면 프롬프트의 예시를 손봐야 한다.

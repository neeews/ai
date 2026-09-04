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

`--from-db` 는 `labeled_by`·`origin` 컬럼을 읽어 사람 라벨과 AI 라벨을 자동으로 갈라 넣는다.
그 컬럼은 백엔드의 `labels/migration_20260903_label_provenance.sql` 로 추가된다
(적용 전 DB에서도 돌아가되, 그때는 `--labeled-by` 값으로 일괄 기록된다).

DB 접속 정보는 백엔드의 `~/backend/.env` 를 그대로 읽는다
(`DB_HOST` `DB_PORT` `DB_USER` `DB_PASSWORD` `DB_NAME`). 환경변수로 덮어쓸 수도 있다.

로컬에서 고친 코드를 서버로 올리려면 `.deploy.env.example` 을 `.deploy.env` 로 복사해
접속 정보를 채운 뒤 `./sync.sh` 를 실행한다 (데이터·모델·venv는 서버에만 둔다).

## 전체 흐름

```
import_labels.py  백엔드 시드 라벨 반입 (학습용) data/labeled/backend_labels.jsonl
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

### 라벨 파일 세 개를 섞지 않는다

| 파일 | 매긴 주체 | 쓰임 |
|---|---|---|
| `data/labeled/auto_labels.jsonl` | Ollama(exaone3.5:2.4b) | train / val |
| `data/labeled/backend_labels.jsonl` | 백엔드 시드 = claude-opus-5 | train / val |
| `data/labeled/labels.jsonl` | **사람만** | **test (정답지)** |

모든 라벨 행은 `origin` 필드를 들고 다니고, `prepare_dataset.py` 는 `origin="human"` 인
행만 test 에 넣는다. 출처가 없는(`unknown`) 행도 test 에서 빠진다 —
**누가 매겼는지 모르는 라벨은 정답지 자격이 없다.**

> **2026-09-03에 실제로 있었던 일.** 백엔드 시드 300건이 `labels.jsonl`(정답지 파일)로
> 그대로 반입돼 사람 라벨 행세를 했다. 그 시드는 claude-opus-5 가 매긴 것이었고
> (백엔드 `c616972` 커밋: `labeled_by=claude-opus-5`, "검수 후 HUMAN 승격 전제"),
> 승격 검수는 이뤄지지 않았다. 24분 뒤 `fdd65b3` 가 `labeled_by` 컬럼을 지우면서
> 출처 기록마저 사라졌다. 그 위에서 잰 아래 수치는 전부 정확도가 아니라
> **"exaone 이 opus 를 얼마나 흉내내나"** 였다. 위의 파일 분리와 `origin` 강제는 이 사고의 대책이다.

## 0. 기존 라벨 반입

백엔드가 이미 매겨 둔 라벨(`article_importance_labels` 테이블 / `labels/importance_seed.csv`)이 있으면
먼저 가져온다. 라벨만 있고 본문이 없으므로 기사 id로 DB에서 본문을 채워 넣는다.
반입한 라벨은 **학습용**(`backend_labels.jsonl`)으로 들어간다.

```bash
python import_labels.py --from-db          # DB 테이블에서
python import_labels.py --csv path/to/importance_seed.csv --dry-run
python import_labels.py --migrate          # 예전에 정답지 파일로 섞여 들어간 라벨 분리
```

DB 테이블에는 라벨 주체 컬럼이 없다(`fdd65b3` 에서 제거됨). 그래서 `--labeled-by` 로
누가 매겼는지 직접 적고, 기본값은 `config.py` 의 `BACKEND_LABELED_BY`(=`claude-opus-5`)다.
사람이 매긴 라벨을 반입할 때만 `--labeled-by human` 을 쓰며, 이때만 `labels.jsonl` 로 들어간다.

`--migrate` 는 예전 `labels.jsonl` 을 훑어 `origin` 이 사람이 아닌 행을 학습용 파일로 옮긴다.
`origin` 이 아예 없는 행은 `unknown` 으로 찍혀 남되 test 에는 들어가지 않는다.

`LOW/MEDIUM/HIGH` → 정수 라벨 변환은 `config.py` 의 `BACKEND_TO_LABEL` 이 담당한다.

## 1. 기사 추출

```bash
python export_db.py --n 3000 --min-len 300
```

카테고리(정치/경제/사회/세계/IT·과학/스포츠/연예문화/종합) 8개에서 균등하게 뽑는다.
`--min-len` 미만의 짧은 기사는 제외 — 본문이 100자도 안 되면 사람도 중요도를 못 정한다.
여러 번 실행해도 이미 뽑은 기사는 건너뛴다.

## 2-a. LLM 자동 라벨링 (학습용)

**먼저 이게 쓸 만한지부터 재라.** 사람 라벨이 조금이라도 있으면 그것에만 자동 라벨을 돌려
일치율을 확인한 뒤에 전체를 돌린다. 안 그러면 6시간 돌리고 못 쓰는 라벨을 얻는다.

```bash
python autolabel.py --only-labeled --strategy cascade --out data/labeled/try1.jsonl
python eval_autolabel.py --auto data/labeled/try1.jsonl --split dev       # 프롬프트 고칠 때
python eval_autolabel.py --auto data/labeled/try1.jsonl --split holdout   # 최종 확인
```

프롬프트는 `dev` 만 보고 고치고 `holdout` 으로 확인한다.
고치면서 같은 데이터로 재면 좋아 보이기만 한다.

### 측정 결과 (exaone3.5:2.4b, 3단계, **claude-opus-5 라벨 100건 기준** / holdout 50건)

> ⚠ 이 표의 기준은 사람이 아니라 claude-opus-5 다. 따라서 아래 숫자는 정확도가 아니라
> **opus 와의 일치율**이다. 사람 정답지가 만들어지면 다시 재고 이 표를 교체해야 한다.
> 프롬프트 방식들의 상대 순위(cascade > 한번에 > 절차설명)는 그래도 참고할 만하다.

| 방식 | opus 라벨과의 일치율 | 비고 |
|---|---|---|
| 등급을 한 번에 물음 | 32% | 우연 수준(33%). 66%를 '중요'로 몰아버림 |
| 판정 절차를 자세히 준 프롬프트 | 26% | 더 나빠짐 — 2.4B 모델은 다단계 채점표를 못 따라감 |
| **예/아니오 두 번 (`--strategy cascade`)** | **44%** | 편향도 사라짐. 작은 모델은 다지선다보다 이분 판단에 강하다 |

`cascade` 는 "지역·개인 한정 소식인가?" → 예면 0, 아니면 "1면 머리기사감인가?" → 예면 2, 아니면 1.


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

## 2-b. 사람 라벨링 (평가용, 150건 권장)

**이 프로젝트에서 사람이 직접 해야 하는 유일한 작업이다.** 여기서 나온 라벨만 정답지가 된다.

방법은 두 가지고, 결과는 같은 곳으로 모인다.

**(a) 웹 관리자 화면** — 백엔드 `/admin/labeling` (ROLE_ADMIN). 폰에서도 매길 수 있다.

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| `GET` | `/admin/labeling/next?size=20&filter=NO_SEED&round=0` | 아직 안 매긴 기사 목록 |
| `POST` | `/admin/labeling/{articleId}` | `{"label":"HIGH","round":0}` 저장 |
| `GET` | `/admin/labeling/stats` | 진행 건수·분포·시드 대조율·자기일치율 |

`filter` 는 `NO_SEED`(정답지용) / `SEED_ONLY`(대조군) / `ALL`. `round=1` 로 다시 매기면
백엔드가 1회차와 대조해 자기일치율을 계산해 준다. 매긴 라벨은 DB에 쌓이므로
`python import_labels.py --from-db` 로 반입한다 — `origin=HUMAN` 인 행만 정답지 파일로 들어간다.

**(b) 터미널 CLI** — 서버에서 바로 돌린다.

```bash
python label.py --exclude-seed --limit 100   # ① 정답지 100건 — 시드에 없는 기사만
python label.py --only-seed --limit 50       # ② 대조군 50건 — 시드에 있는 기사, 시드 라벨은 안 보임
python label.py --stats                      # 분포 확인
```

| 몫 | 건수 | 무엇을 위한 것인가 |
|---|---|---|
| ① 정답지 | 100건 | 모델 채점용. LLM이 손댄 적 없는 기사라 오염이 0이다 |
| ② 대조군 | 50건 | 백엔드 시드(opus 라벨)를 학습에 써도 되는지 확인. 겹치는 기사가 있어야 비교가 된다 |

라벨링 다음날 같은 기사 50건을 다시 매겨 **자기 자신과의 일치율**을 재 둔다.
사람끼리도 안 맞는 일이라면 모델에게 그 이상을 요구할 수 없다 — 성능의 천장이 여기서 나온다.

`--review` 는 LLM 라벨을 먼저 보여준다. 빠르지만 그 숫자에 끌려가므로
**정답지를 만들 때는 쓰지 않는다.** 학습 라벨을 손볼 때만 쓴다.

판정 기준은 `LABELING_GUIDE.md`. 라벨링 중 `g` 를 누르면 그 자리에서 볼 수 있다.
기준이 흔들리면 모델도 흔들리므로, 애매하면 **낮은 등급**으로 통일한다.
각 행에는 그때 적용한 기준 문서의 해시(`guide`)가 함께 저장된다 — 기준을 고치면
그 이전 라벨은 다른 잣대로 매겨진 것이므로, 섞어 쓰기 전에 다시 봐야 한다.

## 3. 데이터셋 분할

```bash
python prepare_dataset.py
```

기본 hybrid 모드: train/val = LLM 라벨(자동 + 백엔드 시드), test = `origin="human"` 인 라벨.
사람이 본 기사는 train/val에서 제외해 평가 오염을 막는다.
같은 기사에 자동 라벨과 시드 라벨이 둘 다 있으면 시드(더 큰 모델) 쪽을 쓴다.

겹치는 기사가 있으면 **자동 라벨 vs 사람**, **시드 vs 사람** 일치율을 각각 출력한다.
앞은 exaone 을 계속 쓸지, 뒤는 opus 시드 300건을 학습에 넣어도 되는지의 근거다.

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
| `import_labels.py` | 백엔드 라벨(LOW/MEDIUM/HIGH) 반입, `--migrate` 로 출처 분리 |
| `LABELING_GUIDE.md` | 등급 판정 기준 |
| `sync.sh` | 로컬 → 서버 코드 동기화 |

## 알아둘 것

- KoELECTRA 체크포인트는 TF에서 변환돼 일부 가중치가 non-contiguous 로 올라온다.
  그대로 두면 저장 단계에서 죽으므로 `train.py` 가 로드 직후 `.contiguous()` 로 정렬한다.
- `requirements.txt` 의 버전은 서로 맞물려 있다. 특히 `accelerate` 가 없으면 `Trainer` 가 뜨지 않는다.
- LLM 자동 라벨은 중요도를 부풀리는 경향이 있다. 학습 전에 반드시 분포를 확인하고,
  0등급이 거의 안 나오면 프롬프트의 예시를 손봐야 한다.

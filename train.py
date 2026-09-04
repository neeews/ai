"""KoELECTRA 를 뉴스 중요도 5단계 분류로 파인튜닝한다.

    python3 train.py                      # 기본 설정 (GPU 자동 감지)
    python3 train.py --epochs 8 --lr 2e-5
    python3 train.py --no-class-weights   # 클래스 가중치 끄기

학습이 끝나면 models/koelectra-importance/ 에 최종 모델과 test 평가 결과가 저장된다.
"""
from __future__ import annotations

import argparse
import inspect
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import (
    BASE_MODEL,
    SERVE_TOP_N,
    BATCH_SIZE_CPU,
    BATCH_SIZE_GPU,
    EPOCHS,
    LABEL_NAMES,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DIR,
    NUM_LABELS,
    SEED,
    SPLIT_DIR,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


class NewsDataset(Dataset):
    """미리 토크나이즈해 두고 인덱스로 꺼내 쓰는 단순 데이터셋."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.labels = [row["label"] for row in rows]
        self.encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """등급별 데이터 수가 불균형할 때 적은 등급에 더 큰 손실 가중치를 준다."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, NUM_LABELS), labels.view(-1), weight=weight
        )
        return (loss, outputs) if return_outputs else loss


def load_split(name: str) -> list[dict]:
    path = SPLIT_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다. 먼저 `python3 prepare_dataset.py` 를 실행하세요.")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }
    if NUM_LABELS == 2:
        # 이진에서는 '중요'로 뽑은 것 중 몇 개가 맞았나(precision)가 서비스 체감을 지배한다.
        # 메인에 SERVE_TOP_N 건만 노출하므로 확률 상위 N건의 정확도를 함께 잰다.
        metrics["precision"] = precision_score(labels, preds, pos_label=1, zero_division=0)
        metrics["recall"] = recall_score(labels, preds, pos_label=1, zero_division=0)
        metrics["f1"] = f1_score(labels, preds, pos_label=1, zero_division=0)
        scores = logits[:, 1] - logits[:, 0]
        n = min(SERVE_TOP_N, len(labels))
        top = np.argsort(-scores)[:n]
        metrics[f"precision_at_{SERVE_TOP_N}"] = float(np.mean(labels[top] == 1)) if n else 0.0
    else:
        # 순서가 있는 등급이라 "1칸 차이"와 "여러 칸 차이"를 구분해야 한다 → QWK
        metrics["qwk"] = cohen_kappa_score(labels, preds, weights="quadratic")
        # 인접 등급까지 맞다고 치는 관대한 정확도 (실사용 체감에 가까움)
        metrics["adjacent_accuracy"] = float(np.mean(np.abs(preds - labels) <= 1))
    return metrics


def class_weights_from(rows: list[dict]) -> torch.Tensor:
    """등급별 빈도의 역수를 가중치로. 없는 등급은 0이 되지 않도록 1로 둔다."""
    counts = np.bincount([r["label"] for r in rows], minlength=NUM_LABELS).astype(float)
    weights = np.where(counts > 0, counts.sum() / (NUM_LABELS * np.maximum(counts, 1)), 1.0)
    return torch.tensor(weights, dtype=torch.float)


def build_training_args(output_dir, batch_size: int, use_fp16: bool, args) -> TrainingArguments:
    """transformers 버전에 따라 인자 이름이 달라서(evaluation_strategy → eval_strategy) 맞춰 넣는다."""
    kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        # 이진에서 macro_f1 은 다수 클래스(0)에 끌려간다. 우리가 신경 쓰는 건 '중요' 쪽이다.
        metric_for_best_model="f1" if NUM_LABELS == 2 else "macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        seed=args.seed,
        report_to=[],
    )
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_key = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    kwargs[eval_key] = "epoch"
    kwargs["save_strategy"] = "epoch"
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in supported})


def main() -> None:
    parser = argparse.ArgumentParser(description="KoELECTRA 중요도 분류 파인튜닝")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=None, help="미지정 시 GPU/CPU 따라 자동")
    parser.add_argument("--no-class-weights", action="store_true", help="클래스 가중치 사용 안 함")
    parser.add_argument("--patience", type=int, default=2, help="early stopping 인내 epoch")
    parser.add_argument("--output", default=str(MODEL_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    from pathlib import Path
    output_dir = Path(args.output)

    has_gpu = torch.cuda.is_available()
    batch_size = args.batch_size or (BATCH_SIZE_GPU if has_gpu else BATCH_SIZE_CPU)
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    print(f"장치: {device_name}  |  batch={batch_size}  max_len={args.max_length}  fp16={has_gpu}")

    train_rows, val_rows, test_rows = load_split("train"), load_split("val"), load_split("test")
    print(f"데이터: train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label={i: LABEL_NAMES[i] for i in range(NUM_LABELS)},
        label2id={LABEL_NAMES[i]: i for i in range(NUM_LABELS)},
    )
    # KoELECTRA 체크포인트는 TF에서 변환돼 일부 가중치가 non-contiguous 로 올라온다.
    # 그대로 두면 safetensors 저장 단계에서 ValueError 로 죽으므로 미리 정렬해 둔다.
    for param in model.parameters():
        param.data = param.data.contiguous()

    datasets = {
        name: NewsDataset(rows, tokenizer, args.max_length)
        for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows))
    }

    weights = None if args.no_class_weights else class_weights_from(train_rows)
    if weights is not None:
        print("클래스 가중치: " + ", ".join(f"{i}:{w:.2f}" for i, w in enumerate(weights.tolist())))

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=build_training_args(output_dir, batch_size, has_gpu, args),
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("\n=== 학습 시작 ===")
    trainer.train()

    print("\n=== test 평가 ===")
    metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    keys = (("test_accuracy", "test_macro_f1", "test_precision", "test_recall", "test_f1",
             f"test_precision_at_{SERVE_TOP_N}") if NUM_LABELS == 2
            else ("test_accuracy", "test_macro_f1", "test_qwk", "test_adjacent_accuracy"))
    for key in keys:
        if key in metrics:
            print(f"  {key:<24} {metrics[key]:.4f}")

    preds = np.argmax(trainer.predict(datasets["test"]).predictions, axis=-1)
    truth = np.array([r["label"] for r in test_rows])
    present = sorted(set(truth.tolist()) | set(preds.tolist()))
    print("\n등급별 성능")
    print(classification_report(
        truth, preds, labels=present,
        target_names=[f"{i} {LABEL_NAMES[i]}" for i in present], zero_division=0,
    ))
    print("혼동 행렬 (행=정답, 열=예측)")
    print(confusion_matrix(truth, preds, labels=present))

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "test_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n모델 저장 완료: {output_dir}")


if __name__ == "__main__":
    main()

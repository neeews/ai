"""학습한 모델로 기사 중요도를 예측한다.

    python3 predict.py --text "제목\n본문..."
    python3 predict.py --id 81397 --id 81396        # DB의 기사 id로
    python3 predict.py --db-latest 20               # 최근 기사 20건
    python3 predict.py --file data/splits/test.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABEL_NAMES, MAX_LENGTH, MODEL_DIR


class ImportanceScorer:
    def __init__(self, model_dir=MODEL_DIR, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        results: list[dict] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt",
            )
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
            for row in probs:
                label = int(torch.argmax(row))
                results.append({
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "confidence": float(row[label]),
                    # 확률 가중 평균 — 0~4 연속값. 정렬·랭킹에는 이쪽이 부드럽다
                    "score": float(sum(i * p for i, p in enumerate(row.tolist()))),
                    "probs": [round(p, 4) for p in row.tolist()],
                })
        return results


def texts_from_db(ids: list[int] | None, latest: int | None) -> list[tuple[int, str, str]]:
    from db import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            if ids:
                placeholders = ",".join(["%s"] * len(ids))
                cur.execute(
                    f"SELECT id, title, description FROM articles WHERE id IN ({placeholders})", ids
                )
            else:
                cur.execute(
                    "SELECT id, title, description FROM articles "
                    "ORDER BY published_at DESC LIMIT %s", (latest,)
                )
            return [
                (int(r["id"]), r["title"] or "", f"{r['title'] or ''}\n{r['description'] or ''}")
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="기사 중요도 예측")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--text", action="append", help="직접 입력한 텍스트")
    parser.add_argument("--id", action="append", type=int, help="DB 기사 id")
    parser.add_argument("--db-latest", type=int, help="최근 기사 N건")
    parser.add_argument("--file", help="jsonl 파일 (text 필드 사용)")
    args = parser.parse_args()

    labels: list[str] = []
    texts: list[str] = []

    if args.text:
        texts.extend(args.text)
        labels.extend(t[:50] for t in args.text)
    if args.id or args.db_latest:
        for aid, title, text in texts_from_db(args.id, args.db_latest):
            labels.append(f"[{aid}] {title[:50]}")
            texts.append(text)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    texts.append(row["text"])
                    labels.append(row.get("title", row["text"][:50]))

    if not texts:
        parser.error("--text / --id / --db-latest / --file 중 하나는 있어야 합니다.")

    scorer = ImportanceScorer(args.model_dir)
    for name, result in zip(labels, scorer.score(texts)):
        print(f"{result['label']} ({result['label_name']:<8}) "
              f"conf {result['confidence']:.2f}  score {result['score']:.2f}  |  {name}")


if __name__ == "__main__":
    main()

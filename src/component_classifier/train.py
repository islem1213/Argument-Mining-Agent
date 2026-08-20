"""Fine-tune a small text classifier from PE/AAEC-style JSONL conversions.

Each row must have ``text`` and one of the canonical labels in LABELS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["major_claim", "claim", "premise", "non_argumentative"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()
    try:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -e .[train]") from exc

    def read(path: Path) -> Dataset:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row["labels"] = LABELS.index(row["label"])
        return Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train, validation = read(args.train).map(tokenize, batched=True), read(args.validation).map(tokenize, batched=True)
    label_map = {i: label for i, label in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=len(LABELS), id2label=label_map, label2id={v: k for k, v in label_map.items()})
    training = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True)
    Trainer(model=model, args=training, train_dataset=train, eval_dataset=validation, processing_class=tokenizer).train()
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()

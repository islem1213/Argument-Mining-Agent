"""Fine-tune a component classifier on JSONL produced by ``argmine.data.brat``.

Each row needs ``text`` and ``label`` (one of LABELS). If rows carry a
``context`` field (the surrounding sentence/paragraph) it is passed as the
second sequence, which typically helps distinguish claims from premises.

    python -m argmine.components.train --train data/processed/components_train.jsonl \
        --validation data/processed/components_test.jsonl --output weights/pe-deberta \
        --base-model microsoft/deberta-v3-base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["major_claim", "claim", "premise", "non_argumentative"]


def macro_f1(predictions: list[int], references: list[int], n_labels: int = len(LABELS)) -> float:
    scores = []
    for label in range(n_labels):
        tp = sum(p == r == label for p, r in zip(predictions, references))
        fp = sum(p == label != r for p, r in zip(predictions, references))
        fn = sum(r == label != p for p, r in zip(predictions, references))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--with-context", action="store_true", help="Use the row's `context` as second segment")
    args = parser.parse_args()
    try:
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -e '.[train]'") from exc
    set_seed(args.seed)

    def read(path: Path) -> Dataset:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return Dataset.from_list([{"text": r["text"], "context": r.get("context", ""),
                                   "labels": LABELS.index(r["label"])} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch: dict) -> dict:
        if args.with_context:
            return tokenizer(batch["text"], batch["context"], truncation="only_second", max_length=256)
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train = read(args.train).map(tokenize, batched=True)
    validation = read(args.validation).map(tokenize, batched=True)
    label_map = dict(enumerate(LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=label_map, label2id={v: k for k, v in label_map.items()})

    def compute_metrics(eval_pred) -> dict:
        logits, refs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": macro_f1(preds.tolist(), refs.tolist()), "accuracy": float((preds == refs).mean())}

    training = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", seed=args.seed, report_to=[],
    )
    trainer = Trainer(model=model, args=training, train_dataset=train, eval_dataset=validation,
                      processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)
    trainer.train()
    print(json.dumps(trainer.evaluate(), indent=2))
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()

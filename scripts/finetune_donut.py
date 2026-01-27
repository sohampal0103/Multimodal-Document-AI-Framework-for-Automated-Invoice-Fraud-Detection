import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    Trainer,
    TrainingArguments,
    VisionEncoderDecoderModel,
)


class JsonlDonutDataset(Dataset):
    def __init__(self, jsonl_path: str) -> None:
        self.jsonl_path = jsonl_path
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

        if not self.items:
            raise ValueError(f"Empty dataset: {jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        image_path = item["image"]
        gt = item["ground_truth"]

        # Support repo-relative paths.
        p = Path(image_path)
        if not p.is_absolute():
            p = Path.cwd() / p

        img = Image.open(p).convert("RGB")
        return {"image": img, "ground_truth": gt}


@dataclass
class DonutCollator:
    processor: DonutProcessor
    task_prompt: str
    max_length: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [x["image"] for x in batch]
        gts = [x["ground_truth"] for x in batch]

        pixel_values = self.processor(images, return_tensors="pt").pixel_values

        # Build target sequences from ground-truth JSON.
        target_texts = [self.processor.json2token(gt) for gt in gts]
        labels = self.processor.tokenizer(
            target_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids

        # Mask padding tokens for loss.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        decoder_input_ids = self.processor.tokenizer(
            [self.task_prompt] * len(batch),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune Donut on an invoice JSONL dataset.")
    p.add_argument("--train", default="data/real/donut/train.jsonl")
    p.add_argument("--val", default="data/real/donut/val.jsonl")
    p.add_argument(
        "--model-id",
        default=os.getenv("DONUT_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2"),
    )
    p.add_argument("--out", default="models/donut-invoice")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument(
        "--task-prompt",
        default="<s_cord-v2>",
        help="Donut task prompt token; choose one matching your pretraining/fine-tuning scheme",
    )
    args = p.parse_args()

    train_ds = JsonlDonutDataset(args.train)
    val_ds = JsonlDonutDataset(args.val)

    processor = DonutProcessor.from_pretrained(args.model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id)

    # Recommended Donut settings.
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(args.task_prompt)
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    collator = DonutCollator(processor=processor, task_prompt=args.task_prompt, max_length=args.max_length)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save in HF format
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    print(f"Saved fine-tuned model to: {out}")


if __name__ == "__main__":
    main()

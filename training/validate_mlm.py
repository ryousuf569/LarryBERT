#!/usr/bin/env python3
"""
MLM validation script (no CLI args).
Edit the PATHS section, then hit Run in VS Code.

Evaluates a saved MLM checkpoint on mlm_val.txt using proper
token concatenation + fixed-length blocks so eval_loss is finite.
Prints eval_loss + perplexity.
"""

import math
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_DIR = r"C:\Users\JrD\Downloads\LARRYBERT\LarryBERT\models\larrybert-base-mlm\checkpoint-15000"
VAL_TXT   = r"C:\Users\JrD\Downloads\LARRYBERT\LarryBERT\data\mlm\mlm_val.txt"
BASE_TOKENIZER_DIR = r"C:\Users\JrD\Downloads\LARRYBERT\LarryBERT\models\larrybert-base-mlm"
OUT_DIR = r"C:\Users\ryous\Downloads\larrybert\models\larrybert-base-mlm-final"
BLOCK_SIZE = 128
BATCH_SIZE = 4
MLM_PROB = 0.15


def main() -> None:
    model_dir = Path(MODEL_DIR)
    val_txt = Path(VAL_TXT)

    if not model_dir.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {model_dir}")
    if not val_txt.exists():
        raise FileNotFoundError(f"VAL_TXT not found: {val_txt}")

    # Load tokenizer (often NOT saved in checkpoint folders)
    tok_dir = Path(BASE_TOKENIZER_DIR) if BASE_TOKENIZER_DIR else model_dir
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tok_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir), local_files_only=True)

    ds = load_dataset("text", data_files={"validation": str(val_txt)})
    def tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=True, truncation=True)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE  # drop remainder
        if total_len == 0:
            raise ValueError(
                "After concatenation, total token length is 0. "
                "Your VAL file may be empty or tokenization produced no tokens."
            )
        return {
            k: [t[i:i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
            for k, t in concatenated.items()
        }

    lm_val = tokenized.map(group_texts, batched=True)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB
    )

    targs = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        use_cpu=True,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        eval_dataset=lm_val["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    tokenizer.save_pretrained(OUT_DIR)
    model.save_pretrained(OUT_DIR)

    print("\n====================")
    print("MLM VALIDATION RESULT")
    print("====================")
    print("MODEL_DIR:", model_dir)
    print("TOKENIZER_DIR:", tok_dir)
    print("VAL_TXT:", val_txt)
    print("BLOCK_SIZE:", BLOCK_SIZE, "BATCH_SIZE:", BATCH_SIZE, "MLM_PROB:", MLM_PROB)
    print("METRICS:", metrics)

    if eval_loss is not None and math.isfinite(eval_loss):
        print(f"Perplexity (exp(eval_loss)): {math.exp(eval_loss):.3f}")
    else:
        print("eval_loss is not finite (NaN/inf). Try BLOCK_SIZE=64 or BATCH_SIZE=8, or check VAL text.")


if __name__ == "__main__":
    main()
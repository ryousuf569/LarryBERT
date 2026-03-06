from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import os

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "models/larrybert-base-mlm"

TRAIN_FILE = "mlm_train.txt"
VAL_FILE = "mlm_val.txt"

MAX_LENGTH = 128


def get_dataset(train_file, val_file):
    ds = load_dataset("text",data_files={"train": train_file, "validation": val_file,})
    return ds


def tokenize_dataset(ds, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def build_trainer(model, tokenizer, tokenized, output_dir):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # evaluation
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,

        # keep the best checkpoint automatically
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # training length
        num_train_epochs=3,
        max_steps=-1,  # train by epochs instead of hard capping steps

        # optimizer
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,

        # RTX 3060 laptop friendly
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        # GPU settings
        fp16=True,

        # better dataloader behavior
        dataloader_pin_memory=True,

        # reporting
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    return trainer


def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    ds = get_dataset(TRAIN_FILE, VAL_FILE)
    tokenized = tokenize_dataset(ds, tokenizer)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized=tokenized,
        output_dir=OUTPUT_DIR,
    )

    trainer.train()

    # save best/final model locally
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


def train_from(saved_model: str, hub_model_id: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(saved_model)
    model = AutoModelForMaskedLM.from_pretrained(saved_model)

    ds = get_dataset(TRAIN_FILE, VAL_FILE)
    tokenized = tokenize_dataset(ds, tokenizer)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized=tokenized,
        output_dir=OUTPUT_DIR,
        hub_model_id=hub_model_id,
    )

    # resume from checkpoint
    trainer.train(resume_from_checkpoint=saved_model)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

train()
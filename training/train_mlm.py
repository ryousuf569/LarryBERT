from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import re as r

MODEL_NAME = "bert-base-uncased"  # keep BERT for LarryBERT branding

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load .txt files: one line = one example
    ds = load_dataset(
        "text",
        data_files={"train": "mlm_train.txt", "validation": "mlm_val.txt"},
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    args = TrainingArguments(
        output_dir="models/larrybert-base-mlm",
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=2,     # CPU safe
        per_device_eval_batch_size=2,
        num_train_epochs=1,                # start with 1
        max_steps=5000,                    # CAP STEPS for CPU
        warmup_ratio=0.06,
        report_to="none",
        use_cpu=True,                      # force CPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("models/larrybert-base-mlm")
    tokenizer.save_pretrained("models/larrybert-base-mlm")


def train_from(saved_model):
    tokenizer = AutoTokenizer.from_pretrained(saved_model)
    model = AutoModelForMaskedLM.from_pretrained(saved_model, local_files_only=True)

    ds = load_dataset(
        "text",
        data_files={"train": r"C:\Users\ryous\Downloads\larrybert\data\txt_files\train.txt", "validation": r"C:\Users\ryous\Downloads\larrybert\data\txt_files\val.txt"},
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    args = TrainingArguments(
        output_dir="models/larrybert-base-mlm",  
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=3000,
        warmup_steps=500,
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=saved_model)

    trainer.save_model("models/larrybert-base-mlm/second-run")
    tokenizer.save_pretrained("models/larrybert-base-mlm/second-run")

train_from(r"C:\Users\ryous\Downloads\larrybert\models\larrybert-base-mlm\checkpoint-1500")
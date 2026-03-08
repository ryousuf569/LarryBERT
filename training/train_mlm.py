from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from multiprocessing import freeze_support
import torch

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = r"N:\LARRYBERT\LarryBERT\models\larry-bert-base"

TRAIN_FILE = r"N:\LARRYBERT\LarryBERT\data\mlm\mlm_train.txt"
VAL_FILE = r"N:\LARRYBERT\LarryBERT\data\mlm\mlm_val.txt"

BLOCK_SIZE = 128

MASK_EVAL_SENTENCES = [
    "Shai Gilgeous-Alexander is elite at drawing [MASK] on drives.",
    "Stephen Curry knocked down seven [MASK] in the fourth quarter.",
    "Victor Wembanyama protects the [MASK] at an elite level.",
    "The Nuggets stagger Jokic with the second [MASK].",
    "Milwaukee's offensive [MASK] improved after the timeout.",
]

TOP_K = 5


def get_dataset(train_file, val_file):
    return load_dataset(
        "text",
        data_files={
            "train": train_file,
            "validation": val_file,
        },
    )


def tokenize_dataset(ds, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            return_special_tokens_mask=True,
            truncation=False,
            padding=False,
        )

    return ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing text",
    )


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    if total_length == 0:
        return {k: [] for k in concatenated_examples.keys()}

    result = {
        k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    return result


def prepare_mlm_dataset(ds, tokenizer):
    tokenized = tokenize_dataset(ds, tokenizer)
    grouped = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping into {BLOCK_SIZE}-token chunks",
    )

    print("\nDataset sizes after grouping:")
    print(f"  train: {len(grouped['train'])}")
    print(f"  validation: {len(grouped['validation'])}")

    if len(grouped["train"]) == 0:
        raise ValueError("Train dataset is empty after grouping.")

    if len(grouped["validation"]) == 0:
        raise ValueError("Validation dataset is empty after grouping.")

    return grouped


def run_mask_eval(model, tokenizer, sentences, top_k=5):
    model.eval()

    device = next(model.parameters()).device
    mask_token_id = tokenizer.mask_token_id

    print("\n" + "=" * 20 + " MASKED SENTENCE EVAL " + "=" * 20)

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            print(f"\n[SKIP] No [MASK] token found: {sentence}")
            continue

        mask_index = mask_positions[0, 1].item()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, mask_index]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, top_k)

        predictions = []
        for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            token = tokenizer.convert_ids_to_tokens(token_id)
            predictions.append(f"{token} ({prob:.2f})")

        print(f"\n{sentence}")
        print("  -> " + ", ".join(predictions))

    print("=" * 60 + "\n")


class MaskEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, sentences, top_k=5):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.top_k = top_k

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control

        print(f"\n[MaskEvalCallback] Running masked eval at step {state.global_step}...")
        run_mask_eval(
            model=model,
            tokenizer=self.tokenizer,
            sentences=self.sentences,
            top_k=self.top_k,
        )
        return control


def build_trainer(model, tokenizer, tokenized, output_dir):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=5000,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            MaskEvalCallback(
                tokenizer=tokenizer,
                sentences=MASK_EVAL_SENTENCES,
                top_k=TOP_K,
            )
        ],
    )
    return trainer


def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    ds = get_dataset(TRAIN_FILE, VAL_FILE)
    tokenized = prepare_mlm_dataset(ds, tokenizer)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized=tokenized,
        output_dir=OUTPUT_DIR,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


def train_from(saved_model: str):
    tokenizer = AutoTokenizer.from_pretrained(saved_model, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(saved_model)

    ds = get_dataset(TRAIN_FILE, VAL_FILE)
    tokenized = prepare_mlm_dataset(ds, tokenizer)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized=tokenized,
        output_dir=OUTPUT_DIR,
    )

    trainer.train(resume_from_checkpoint=saved_model)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    freeze_support()
    train()
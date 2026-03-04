# LarryBERT
**A BERT-style language model fine-tuned on NBA text (work in progress).** :contentReference[oaicite:1]{index=1}

## What this is
LarryBERT is my ongoing project to fine-tune **BERT (bert-base-uncased)** using **Masked Language Modeling (MLM)** on an NBA-focused text corpus, using the Hugging Face **Trainer** pipeline. :contentReference[oaicite:2]{index=2}

The goal is to build a model that understands NBA-specific wording (players, teams, slang, game events) better than a general English model.

## Current progress
- ✅ Dataset prep scripts (auto-labeling + splitting)
- ✅ MLM training pipeline (CPU-friendly settings)
- ✅ Validation script + perplexity reporting
- ✅ Simple “fill-mask” comparison script vs baseline BERT :contentReference[oaicite:3]{index=3}

### Latest checkpoint result
**Checkpoint:** `models/larrybert-base-mlm/checkpoint-1500`  
- `eval_loss`: **1.3040**
- `perplexity`: **3.684** :contentReference[oaicite:4]{index=4}

## Repo structure
- `dataprep/` — dataset prep utilities (`larrybert_autolabel.py`, `randomsplit.py`) :contentReference[oaicite:5]{index=5}  
- `data/` — raw/processed text & json files :contentReference[oaicite:6]{index=6}  
- `training/` — training + validation scripts (`train_mlm.py`, `validate_mlm.py`) :contentReference[oaicite:7]{index=7}  
- `comparison/` — quick baseline comparison (`compare_model.py`) :contentReference[oaicite:8]{index=8}  
- `models/` — saved model outputs/checkpoints (kept small / limited checkpoints) :contentReference[oaicite:9]{index=9}  

## Sources / References
- https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling
- https://huggingface.co/google-bert/bert-base-uncased
- https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
- https://huggingface.co/blog/bert-cpu-scaling-part-1

## Quickstart (local)
> Assumes you have Python 3.10+ and a working PyTorch install.

```bash
pip install -r requirements.txt
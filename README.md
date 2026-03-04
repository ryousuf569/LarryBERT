Sources

https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling
https://huggingface.co/google-bert/bert-base-uncased
https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments

HUGE HELP in training bert on CPU info
https://huggingface.co/blog/bert-cpu-scaling-part-1


====================
CHECKPOINT-1500 MLM VALIDATION RESULT
====================
MODEL_DIR: larrybert-base-mlm\checkpoint-1500
TOKENIZER_DIR: larrybert-base-mlm
VAL_TXT: mlm_val.txt
BLOCK_SIZE: 128 BATCH_SIZE: 4 MLM_PROB: 0.15
METRICS: {'eval_loss': 1.304042935371399, 'eval_model_preparation_time': 0.016, 'eval_runtime': 925.0004, 'eval_samples_per_second': 3.181, 'eval_steps_per_second': 0.796}
Perplexity (exp(eval_loss)): 3.684
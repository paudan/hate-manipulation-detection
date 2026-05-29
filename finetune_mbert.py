import os
import gc
import numpy as np
import torch
from torch import nn
from datasets import DatasetDict
from unsloth import FastModel, is_bfloat16_supported
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import set_seed
from transformers.trainer_callback import EarlyStoppingCallback
from peft import TaskType
from sklearn.metrics import classification_report
from experiment import preprocess_text
from finetune import split_dataset, compute_metrics

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
full_finetuning = True


def get_training_args(output_dir, batch_size=8, num_epochs=20, eval_batch_size=128, report_to=['tensorboard']):
    return TrainingArguments(
        output_dir=output_dir,
        metric_for_best_model='eval_accuracy',
        load_best_model_at_end=True,
        greater_is_better=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        logging_strategy='epoch',
        log_level='info',
        logging_first_step=True,
        save_strategy='epoch',
        save_total_limit=2,
        num_train_epochs=num_epochs,
        auto_find_batch_size=False,
        ignore_data_skip=True,
        disable_tqdm=False,
        # overwrite_output_dir=True,
        optim='adamw_8bit',
        learning_rate=5e-5 if full_finetuning else 2e-4,
        gradient_accumulation_steps=1,
        # lr_scheduler_type="cosine",
        fp16_full_eval=False,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to=report_to,
        seed=SEED,
        data_seed=SEED,
        average_tokens_across_devices=False
        
    )

def train_eval_model(model_name, dataset, labels_map, use_lora=True, **training_args):
    set_seed(SEED)
    dataset_dict = split_dataset(dataset)
    id2label = {v: k for k, v in labels_map.items()}
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=2048,
        dtype=dtype,
        num_labels=len(id2label),
        full_finetuning=full_finetuning,
        id2label=id2label,
        label2id=labels_map,
        load_in_4bit=load_in_4bit,
    )
    rank = 32   # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    model = FastModel.get_peft_model(
        model,
        r=rank,
        target_modules="all-linear",
        modules_to_save=["classifier"],
        lora_alpha=rank,
        lora_dropout=0, # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
        random_state=SEED,
        use_rslora=True,  # supports rank stabilized LoRA
        loftq_config=None, 
        task_type=TaskType.SEQ_CLS
    )
    trainer = Trainer(
        model=model,
        args=get_training_args(**training_args),
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['valid'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )
    trainer.train()
    predictions = trainer.predict(dataset_dict['test'])
    print(classification_report(predictions.label_ids, np.argmax(predictions.predictions, axis=1), target_names=labels_map.keys()))
    try:
        # Move model to CPU before deleting to ensure GPU memory is freed
        if torch.cuda.is_available():
            try:
                model.cpu()
            except Exception:
                pass
        del trainer, model, tokenizer
    finally:
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
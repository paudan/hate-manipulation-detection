import os
import gc
from typing import Union, Optional, Any
import numpy as np
import torch
from torch import nn
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers import set_seed
from transformers import TrainingArguments, Trainer 
from transformers.trainer_callback import EarlyStoppingCallback      
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.utils import compute_class_weight
from experiment import preprocess_text

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
SEED = 42


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        labels = inputs.pop("labels", None)
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        logits = outputs.logits
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float, device=logits.device)
        else:
            weight = None
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def split_dataset(ds, train_size=0.7, valid_size=0.15):
    ds = ds.class_encode_column("labels")
    train_test_split = ds.train_test_split(train_size=train_size, shuffle=True, seed=SEED, stratify_by_column='labels')
    train_dataset = train_test_split["train"].shuffle(seed=SEED)
    test_splits = train_test_split["test"].train_test_split(train_size=valid_size/(1-train_size), shuffle=True, seed=SEED, stratify_by_column='labels')
    valid_dataset = test_splits["train"].shuffle(seed=SEED)
    test_dataset = test_splits["test"].shuffle(seed=SEED)
    return DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    })
    
def get_training_args(output_dir, batch_size=64, num_epochs=20, eval_batch_size=128, report_to=['tensorboard']):
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
        # lr_scheduler_type="cosine",
        fp16_full_eval=False,
        fp16=False,
        # fp16_opt_level='O1',
        report_to=report_to,
        seed=SEED,
        data_seed=SEED
    )

def compute_metrics(eval_preds, average='macro'):
    logits, actual = eval_preds
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predictions, average=average, zero_division=0, pos_label=1)
    try:
        roc_auc = roc_auc_score(predictions, actual)
    except:
        roc_auc = np.nan
    return {
       'accuracy': accuracy_score(predictions, actual),
       'f1_score': f1,
       'precision': precision,
       'recall': recall,
       'roc_auc': roc_auc
    }


def get_class_weights(labels):
    labels = [int(x.cpu().item()) if isinstance(x, torch.Tensor) else int(x) for x in labels]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return class_weights


def train_eval_model(model_name, dataset, labels_map, use_lora=True, use_class_weights=False, **training_args):
    set_seed(SEED)
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    processor = AutoModel.from_pretrained(model_name, device_map=device)
    dataset_dict = split_dataset(dataset)
    id2label = {v: k for k, v in labels_map.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=AutoConfig.from_pretrained(model_name, label2id=labels_map, id2label=id2label),
        device_map=device
    )
    if use_lora:
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            modules_to_save=["classifier"], 
            init_lora_weights="olora", 
            target_modules="all-linear"
        )
        model = get_peft_model(model, config)
    trainer_params = dict(
        model=model,
        args=get_training_args(**training_args),
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['valid'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ]        
    )  
    if use_class_weights:
        trainer = WeightedTrainer(
            class_weights=get_class_weights(dataset_dict['train']['labels']),
            **trainer_params
        )
    else:
        trainer = Trainer(**trainer_params)
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
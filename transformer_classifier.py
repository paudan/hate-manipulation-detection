import os
from functools import partial
from typing import Optional
import numpy as np
from datasets import Dataset
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, BertConfig, BertModel
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ['WANDB_DISABLED'] = 'true'
SEED = 42


class TransformerClassifier(BertModel):
    config_class=BertConfig

    def __init__(self, config, num_labels=2, dropout=0.1, use_layer_norm=False, tuned_layers_count=0):
        super().__init__(config)
        self.use_layer_norm = use_layer_norm
        self.num_labels = num_labels
        self.config.problem_type = "single_label_classification"
        # tuned_layers_count = -1 means fine-tune full model
        if tuned_layers_count > -1:
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze last frozen_layers_count layers for finetuning
            if tuned_layers_count > 0:
                for idx in range(-1, -tuned_layers_count-1, -1):
                    for param in self.encoder.layer[idx].parameters():
                        param.requires_grad = True
        if use_layer_norm:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  
    ):
        outputs = super().forward(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        x = outputs[1]
        if self.use_layer_norm:
            x = self.layernorm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        if labels is None:
            return torch.sigmoid(logits)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    

def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def get_training_args(output_dir, batch_size=8, num_epochs=20):
    return TrainingArguments(
        output_dir=output_dir,
        metric_for_best_model='eval_accuracy',
        load_best_model_at_end=True,
        greater_is_better=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        logging_strategy='epoch',
        log_level='info',
        logging_first_step=True,
        save_strategy='epoch',
        save_total_limit=2,
        num_train_epochs=num_epochs,
        auto_find_batch_size=False,
        ignore_data_skip=True,
        disable_tqdm=False,
        overwrite_output_dir=True,
        # lr_scheduler_type="cosine",
        fp16_full_eval=False,
        fp16=False,
        fp16_opt_level='O1',
        report_to=['tensorboard'],
        seed=SEED,
        data_seed=SEED
    )

def compute_metrics(eval_preds, average='binary'):
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

def input_generator(texts, targets):
    for txt, label in zip(texts, targets):
        yield {"text": txt, "labels": label}

def calculate_scores(actual, predictions, average='binary'):
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
       'kappa': cohen_kappa_score(predictions, actual),
       'roc_auc': roc_auc
    }    

def evaluate_fold(predictions, average='binary'):
    actual = predictions.label_ids
    predictions = np.argmax(predictions.predictions, axis=-1)
    return calculate_scores(actual, predictions, average=average)


def cross_validate(texts, targets, model_path, num_folds, cache_dir=None, output_dir='test-classifier', 
                   batch_size=16, num_epochs=20, average='binary', tuned_layers_count=0, 
                   dropout=0.1, use_lora=False, lora_args={}, model_args={}):
    set_seed()
    all_predictions = list()
    all_logits = list()
    all_ground_truth = list()
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    texts = np.array(texts)
    all_valid_results = dict()
    for ind, (train_index, test_index) in enumerate(skf.split(texts, targets)):
        X_train, y_train = texts[train_index], targets[train_index]
        X_test, y_test = texts[test_index], targets[test_index]
        train_dst = Dataset.from_generator(lambda: input_generator(X_train, y_train)).train_test_split(test_size=0.2, shuffle=True, seed=SEED)
        tokenized_dataset = train_dst.map(tokenize_function, batched=True)
        train_dataset = tokenized_dataset["train"].shuffle(seed=SEED)
        valid_dataset = tokenized_dataset["test"].shuffle(seed=SEED)
        eval_dst = Dataset.from_generator(lambda: input_generator(X_test, y_test))
        eval_dataset = eval_dst.map(tokenize_function, batched=True)
        tuned_layers_count = tuned_layers_count if not use_lora else 0
        model=TransformerClassifier.from_pretrained(
            model_path,
            config=AutoConfig.from_pretrained(model_path, cache_dir=cache_dir),
            num_labels=len(set(targets)),
            cache_dir=cache_dir,
            device_map='cuda' if torch.cuda.is_available() else 'cpu',
            tuned_layers_count=tuned_layers_count,
            dropout=dropout                
        )
        if use_lora:
            default_lora_args = {
                "task_type": TaskType.SEQ_CLS,
                "modules_to_save": ["classifier"]
            }
            lora_args.update(default_lora_args)
            config = LoraConfig(**lora_args)
            model = get_peft_model(model, config)
        trainer = Trainer(
            model=model,
            args=get_training_args(output_dir=output_dir, batch_size=batch_size, num_epochs=num_epochs),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=partial(compute_metrics, average=average),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        trainer.train()
        predictions = trainer.predict(eval_dataset)
        all_logits.append(predictions.predictions)
        all_predictions.extend(np.argmax(predictions.predictions, axis=-1))
        all_ground_truth.extend(y_test)
        all_valid_results[ind] = evaluate_fold(predictions, average=average)
        del trainer.model
    all_logits = np.concatenate(all_logits)
    return all_predictions, all_ground_truth, all_logits, all_valid_results

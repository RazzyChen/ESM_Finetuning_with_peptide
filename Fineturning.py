import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EsmConfig,
    TrainingArguments,
    Trainer,
    EsmForSequenceClassification,
    EarlyStoppingCallback,
)
from safetensors.torch import save_file as save_safetensors
import torch
import evaluate
import numpy as np
import wandb
import os
from typing import Dict, List, Any, Union, Tuple

wandb.init(project="ESM2_8M_FT_with_peptide")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载数据
train_file_path: str = "./data/train.csv"
eval_file_path: str = "./data/test.csv"
df_train: pd.DataFrame = pd.read_csv(train_file_path)
df_test: pd.DataFrame = pd.read_csv(eval_file_path)
dataset_train: Dataset = Dataset.from_pandas(df_train)
dataset_test: Dataset = Dataset.from_pandas(df_test)

# 加载预训练模型和 tokenizer
model_name: str = "facebook/esm2_t6_8M_UR50D"
config: EsmConfig = EsmConfig.from_pretrained(
    model_name, position_embedding_type="rotary", num_labels=2
)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
model: EsmForSequenceClassification = EsmForSequenceClassification.from_pretrained(
    model_name, config=config, cache_dir="./model/"
)

for param in model.parameters():
    param.requires_grad = False

# 解冻最后两层
for param in model.classifier.parameters():
    param.requires_grad = True


# 数据预处理
def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[Any]]:
    return tokenizer(
        examples["sequence"], truncation=True, padding="max_length", max_length=20
    )


tokenized_train_datasets: Dataset = dataset_train.map(preprocess_function, batched=True)
tokenized_test_datasets: Dataset = dataset_test.map(preprocess_function, batched=True)

# 训练参数配置
training_args: TrainingArguments = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    seed=42,
    dataloader_num_workers=24,
    logging_dir="./logs",
    fp16=True,
    push_to_hub=False,
    report_to="wandb",
    save_safetensors=True,
    greater_is_better=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    warmup_steps=20,
    lr_scheduler_type="cosine",
)

early_stopping_callback: EarlyStoppingCallback = EarlyStoppingCallback(
    early_stopping_patience=5, early_stopping_threshold=0.001
)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    accuracy_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    return {"accuracy": accuracy["accuracy"], "recall": recall["recall"]}


class MyTrainer(Trainer):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        metrics: Dict[str, float] = None,
        **kwargs: Any,
    ) -> None:
        super().on_evaluate(args, state, control, metrics, **kwargs)
        if metrics:
            print(f"Evaluation Results at step {state.global_step}:")
            print(f"Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            print(f"Recall: {metrics.get('eval_recall', 0):.4f}")


# Trainer
trainer: MyTrainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_test_datasets,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# 训练模型
trainer.train()
trainer.evaluate(tokenized_test_datasets)

# 保存模型为 safetensors 格式
model_path: str = "./Weight"
trainer.save_model(model_path)

# 将权重保存为 safetensors 格式
model_state_dict: Dict[str, torch.Tensor] = model.state_dict()
save_safetensors(model_state_dict, model_path + "/esm2_t6_8M_UR50D_FT.safetensors")

# 保存 tokenizer
tokenizer.save_pretrained(model_path)
print("Model Saved!")

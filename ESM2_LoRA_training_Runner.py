import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EsmConfig,
    TrainingArguments,
    Trainer,
    EsmForSequenceClassification,
)
from peft import AdaLoraConfig, get_peft_model
from safetensors.torch import save_file as save_safetensors
import torch
import evaluate
import numpy as np
import wandb
import os

wandb.init(project="ESM2_adaLoRA_with_peptide")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载数据
train_file_path = "./data/train.csv"
eval_file_path = "./data/test.csv"
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(eval_file_path)
dataset_train = Dataset.from_pandas(df_train)
dataset_test = Dataset.from_pandas(df_test)

# 加载预训练模型和 tokenizer
# model_name = "facebook/esm2_t48_15B_UR50D"
# model_name = "facebook/esm2_t36_3B_UR50D"
# model_name = "facebook/esm2_t33_650M_UR50D"
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForSequenceClassification.from_pretrained(
    model_name, num_labels=2, cache_dir="./model/"
)


# 数据预处理
def preprocess_function(examples):
    return tokenizer(
        examples["sequence"], truncation=True, padding="max_length", max_length=20
    )


tokenized_train_datasets = dataset_train.map(preprocess_function, batched=True)
tokenized_test_datasets = dataset_test.map(preprocess_function, batched=True)

# 配置 AdaLoRA 微调
lora_config = AdaLoraConfig(target_modules=["dense"], lora_dropout=0.7, bias="none")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 训练参数配置
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    seed=42,
    dataloader_num_workers=24,
    logging_dir="./logs",
    fp16=True,
    push_to_hub=False,
    report_to="wandb",
    save_safetensors=True,
    greater_is_better=True,
    evaluation_strategy="epoch",
    warmup_steps=100,
    eval_steps=100,
)


def compute_metrics(eval_pred):
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
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, metrics, **kwargs)
        if metrics:
            print(f"Evaluation Results at step {state.global_step}:")
            print(f"Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            print(f"Recall: {metrics.get('eval_recall', 0):.4f}")


# Trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_test_datasets,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()
trainer.evaluate(tokenized_test_datasets)


# 保存模型为 safetensors 格式
model_path = "./Weight"
trainer.save_model(model_path)

# 将权重保存为 safetensors 格式
model_state_dict = model.state_dict()
save_safetensors(model_state_dict, model_path + "/esm2_t6_8M_UR50D_LORA.safetensors")

# 保存 tokenizer
tokenizer.save_pretrained(model_path)

print("Model Saved!")

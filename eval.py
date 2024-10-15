import torch
from transformers import AutoTokenizer, EsmForSequenceClassification
from safetensors.torch import load_file as load_safetensors
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

# 模型和 tokenizer 路径
base_model_name = "facebook/esm2_t6_8M_UR50D"
safetensor_model_path = "./Weight/esm2_t6_8M_UR50D_FT.safetensors"

# 加载base模型（不加载分类器的权重）
base_model = EsmForSequenceClassification.from_pretrained(
    base_model_name, 
    num_labels=2, 
    ignore_mismatched_sizes=True  # 忽略分类器层大小不匹配
)

# 加载 safetensors 权重
state_dict = load_safetensors(safetensor_model_path)

# 加载所有权重，包括分类器部分
base_model.load_state_dict(state_dict, strict=False)

# 设置模型为评估模式
base_model.eval()

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 加载测试数据
test_data = load_dataset("csv", data_files="./data/test.csv")["train"]

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["sequence"], truncation=True, padding=True, max_length=20)

# 预处理数据集
test_data = test_data.map(preprocess_function, batched=True)

# 确保数据是 PyTorch 张量
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 创建 DataLoader
dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 进行推理和计算指标
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

all_predictions = []
all_labels = []

for batch in tqdm(dataloader):
    # 将输入数据移动到设备
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}  # 排除标签
    labels = batch["labels"].to(device)

    # 推理
    with torch.no_grad():
        outputs = base_model(**inputs)
    
    # 计算预测
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    all_predictions.extend(predictions)
    all_labels.extend(labels.cpu().numpy())

# 计算指标
accuracy = accuracy_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

# 加载原始数据
df = pd.read_csv("./data/test.csv")

# 将预测结果加入到 DataFrame 中
df['pred'] = all_predictions

# 将结果保存到 CSV 文件
df.to_csv("./data/test_pred.csv", index=False)

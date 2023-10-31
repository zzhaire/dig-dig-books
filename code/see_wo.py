import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset

# 载入预训练的BERT模型和分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=24)  # 假设有24个类别

# 准备数据（示例）
texts = ["Your text data goes here.", "Another text sample."]
labels = [0, 1]  # 用于示例的标签，根据你的数据集修改

# 对文本数据进行分词和编码
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 创建数据集和数据加载器
input_ids = encoded_texts["input_ids"]
attention_mask = encoded_texts["attention_mask"]
labels = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_mask, labels)

# 定义训练参数
batch_size = 16
learning_rate = 1e-5
num_epochs = 3

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 模型微调
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 保存微调后的模型
model.save_pretrained("bert_text_classification")

# 推理和预测
model.eval()
with torch.no_grad():
    new_texts = ["New text samples for prediction.", "More text samples."]
    encoded_new_texts = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_new_texts["input_ids"]
    attention_mask = encoded_new_texts["attention_mask"]
    outputs = model(input_ids, attention_mask=attention_mask)
    predicted_labels = torch.argmax(outputs.logits, dim=1)

print("Predicted labels:", predicted_labels)

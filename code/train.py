import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.optim import Adam
from tqdm import tqdm
import torch
import os
import logging

# **************读取数据和模型************
data = pd.read_csv("../dataset/train.csv")
data_part = data.sample(n=60000, random_state=42, replace=True)
data_shuffled = data_part.sample(frac=1, random_state=42)  # 随机打乱数据
train_data, test_data = train_test_split(
    data_shuffled, test_size=0.3, random_state=42
)  # 分割成训练集和测试集

K_FOLDS = 6  # K折训练
# K折训练的模型
kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# ***************下载模型*****************
if 1:# 下载模型
    print("下载模型中...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.save_pretrained("../model/Tokenizer")
    bert = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
    bert.save_pretrained("../model/BERT_ROW")
    bert_basic = BertModel.from_pretrained("bert-base-cased")
    bert_basic.save_pretrained("../model/BERT_BASIC")
    print("!模型下载结束")
if 0:# 
    print("模型加载中...")
    tokenizer = AutoTokenizer.from_pretrained("../model/Tokenizer")
    bert = AutoModelForMaskedLM.from_pretrained("../model/BERT_ROW")
    bert_basic = BertModel.from_pretrained("../model/BERT_BASIC")
    print("模型加载完毕...")
# ***************常量和定义的类与函数************
LABELS = {
    "Literature & Fiction": 0,
    "Animals": 1,
    "Growing Up & Facts of Life": 2,
    "Humor": 3,
    "Cars, Trains & Things That Go": 4,
    "Fairy Tales, Folk Tales & Myths": 5,
    "Activities, Crafts & Games": 6,
    "Science Fiction & Fantasy": 7,
    "Classics": 8,
    "Mysteries & Detectives": 9,
    "Action & Adventure": 10,
    "Geography & Cultures": 11,
    "Education & Reference": 12,
    "Arts, Music & Photography": 13,
    "Holidays & Celebrations": 14,
    "Science, Nature & How It Works": 15,
    "Early Learning": 16,
    "Biographies": 17,
    "History": 18,
    "Children's Cookbooks": 19,
    "Religions": 20,
    "Sports & Outdoors": 21,
    "Comics & Graphic Novels": 22,
    "Computers & Technology": 23,
}

# 日志文件输出目录
logging.basicConfig(filename="../log/train.log", level=logging.INFO)


# ***  封装类 方便数据类型转换 ***************
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [LABELS[label] for label in df["category"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = bert_basic
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 24)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        print("使用gpu")
        model = model.to(device)
        criterion = criterion.to(device)

    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            train_label = train_label.to(torch.long)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0

        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                val_label = val_label.to(torch.long)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        logging.info(
            "\n| Epochs: %d \n| Train Loss: %.3f \n| Train Accuracy: %.3f \n| Val Loss: %.3f \n| Val Accuracy: %.3f \n",
            epoch_num + 1,
            total_loss_train / len(train_data),
            total_acc_train / len(train_data),
            total_loss_val / len(val_data),
            total_acc_val / len(val_data),
        )


# ************** 运行部分 ********************

model = BertClassifier()
model.load_state_dict(torch.load("../model/BERT-1"))

learning_rate = 5e-6  # 设置学习率
epochs = 1  # 设置训练轮数

train(model, train_data, test_data, learning_rate, epochs)

torch.save(model.state_dict(), "../model/BERT-1")

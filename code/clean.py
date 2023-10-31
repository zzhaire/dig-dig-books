import pandas as pd

# 读取CSV文件
data_train = pd.read_csv("../dataset/train.csv")

# 定义函数来提取标题和描述
def extract_title_and_description(text):
    segments = text.split(";")
    title = ""
    description = ""
    for segment in segments:
        if "Title:" in segment:
            title = segment.replace("Title:", "").strip()
        elif "Description:" in segment:
            description = segment.replace("Description:", "").strip()
    return title, description

# 创建新的列来存储标题和描述
data_train['Title'], data_train['Description'] = zip(*data_train['text'].apply(extract_title_and_description))

# 打印前几行以检查结果
print(data_train[['Title', 'Description']].head())

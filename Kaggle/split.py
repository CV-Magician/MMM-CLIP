import json
from sklearn.model_selection import train_test_split

# 读取label.json文件
with open("label.json", "r") as f:
    labels = json.load(f)

# 划分数据集
train_labels, test_labels = train_test_split(labels, test_size=0.9, random_state=42)

# 将划分后的数据保存为train.json和test.json
with open("train.json", "w") as train_file:
    json.dump(train_labels, train_file)

with open("test.json", "w") as test_file:
    json.dump(test_labels, test_file)

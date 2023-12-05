import torch
import clip
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
class_list = [
    "motorcycle",
    "truck",
    "boat",
    "bus",
    "cycle",
    "sitar",
    "ektara",
    "flutes",
    "tabla",
    "harmonium",
]


def calculate_ap(gt, pred):
    precision, recall, _ = precision_recall_curve(gt, pred)
    ap = auc(recall, precision)
    return ap


def calculate(label_data, pred_data, threshold=0.5):
    ap_list = []

    for _, class_name in enumerate(class_list):
        gt_list = [int(sample[class_name]) for sample in label_data]
        pred_list = [1 if sample[class_name] > threshold else 0 for sample in pred_data]

        ap = calculate_ap(gt_list, pred_list)
        ap_list.append(ap)
    print(ap_list)
    mAP = np.mean(ap_list)

    return mAP


prompt_list = []
pred = []
for item in class_list:
    prompt = "A photo of " + item
    prompt_list.append(prompt)

text = clip.tokenize(prompt_list).to(device)


class CustomDataset(Dataset):
    def __init__(self, json_file, image_folder):
        self.data = json.load(open(json_file))
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["Image_Name"]

    def get_label(self, idx):
        item = self.data[idx]
        label = []
        for class_item in class_list:
            label.append(item[class_item])
        return torch.tensor(label)

    def get_image_features(self, idx):
        item = self.data[idx]
        image_name = item["Image_Name"]
        image_path = os.path.join(self.image_folder, image_name)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        features = torch.cat([image_features[0], text_features[0]], dim=-1)
        return features


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.fc.weight.dtype)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


train_dataset = CustomDataset(json_file="train.json", image_folder="images")
test_dataset = CustomDataset(json_file="test.json", image_folder="images")
input_size = 1024
output_size = len(class_list)

mlp = MLP(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    mlp.train()
    train_loss = 0.0
    for idx in tqdm(range(len(train_dataset)), desc="Getting image features"):
        features = train_dataset.get_image_features(idx)
        torch.set_printoptions(profile="full")
        label = train_dataset.get_label(idx).to(torch.float32).to(device)
        output = mlp(features)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
        if (idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(train_dataset)}], Loss: {train_loss/(idx+1)}"
            )
    pred = []
    mlp.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Testing"):
            ans = {}
            features = test_dataset.get_image_features(idx)
            torch.set_printoptions(profile="full")
            output = mlp(features).tolist()
            ans["Image_Name"] = test_dataset[idx]
            for i in range(len(class_list)):
                ans[class_list[i]] = output[i]
            pred.append(ans)
    with open("pred.json", "w", encoding="utf-8") as f:
        json.dump(pred, f)

    with open("test.json", "r") as f:
        label_data = json.load(f)

    with open("pred.json", "r") as f:
        pred_data = json.load(f)

    sorted_label = sorted(label_data, key=lambda x: x["Image_Name"])
    sorted_pred = sorted(pred_data, key=lambda x: x["Image_Name"])

    mAP = calculate(sorted_label, sorted_pred)

    print(f"mAP: {mAP}")

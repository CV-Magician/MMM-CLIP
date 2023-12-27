import torch
import clip
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

folder_path = "images"

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
        return item["name"]

    def get_image_name(self, idx):
        item = self.data[idx]
        image_name = item["Image_Name"]
        return image_name


dataset = CustomDataset(json_file="label.json", image_folder="images")


for idx in tqdm(range(len(dataset)), desc="Processing images"):
    filename = dataset.get_image_name(idx)
    pred_img = {}
    pred_img["Image_Name"] = filename
    file_path = os.path.join(folder_path, filename)
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        sorted_indices = np.argsort(probs[0])[::-1]
        prompt = ""
        prompts = []
        for item in sorted_indices:
            prompt += class_list[int(item)]
            prompts.append(prompt)
            prompt += ", "
        prompts.append("A photo without anything")
        new_text = clip.tokenize(prompts).to(device)
        new_logits, logits_per_text = model(image, new_text)
        new_probs = new_logits.softmax(dim=-1).cpu().numpy()
        pred_img["prompt"] = prompts[int(np.argmax(new_probs[0]))]
    pred.append(pred_img)

for item in pred:
    prompt = item.get("prompt", "")  

    for class_name in class_list:
        if class_name in prompt:
            item[class_name] = 1

        else:
            item[class_name] = 0

with open("pred.json", "w", encoding="utf-8") as file:
    json.dump(pred, file, ensure_ascii=False, indent=2)

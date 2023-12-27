import torch
import clip
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from itertools import combinations

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
all_combinations = []
for r in range(1, len(class_list) + 1):
    all_combinations.extend(combinations(class_list, r))

for item in all_combinations:
    prompt = "A photo of "
    for i in range(len(item) - 1):
        prompt += "a " + item[i] + ","
    if len(item) > 1:
        prompt += "and a " + item[len(item) - 1]
    else:
        prompt += "a " + item[len(item) - 1]
    prompt_list.append(prompt)

prompt_list.append("A photo without anything")
text = clip.tokenize(prompt_list).to(device)

for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        pred_img = {}
        pred_img["Image_Name"] = filename
        file_path = os.path.join(folder_path, filename)
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            pred_img["prompt"] = prompt_list[int(np.argmax(probs[0]))]
        pred.append(pred_img)

for item in pred:
    prompt = item.get("prompt", "")  # 获取'prompt'项的值，默认为空字符串

    for class_name in class_list:
        if class_name in prompt:
            item[class_name] = 1

        else:
            item[class_name] = 0

with open("pred.json", "w", encoding="utf-8") as file:
    json.dump(pred, file, ensure_ascii=False, indent=2)

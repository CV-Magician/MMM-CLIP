import torch
import clip
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np

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

yn_list = []
pred = []
for c in class_list:
    yes_no = []
    yes_no.append("A photo without a" + c)
    yes_no.append("A photo of a" + c)
    yn_list.append(yes_no)
text_list = []
for item in yn_list:
    text_list.append(clip.tokenize(item).to(device))


for filename in tqdm(os.listdir(folder_path), desc="Processing images"):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        pred_img = {}
        pred_img["Image_Name"] = filename
        file_path = os.path.join(folder_path, filename)
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            for i, text in enumerate(text_list):
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                pred_img[class_list[i]] = int(np.argmax(probs[0]))
        pred.append(pred_img)

with open("pred.json", "w", encoding="utf-8") as f:
    json.dump(pred, f)

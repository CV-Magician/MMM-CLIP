import json
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

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


def calculate(label_data, pred_data):
    ap_list = []

    for _, class_name in enumerate(class_list):
        gt_list = [sample[class_name] for sample in label_data]
        pred_list = [sample[class_name] for sample in pred_data]

        ap = calculate_ap(gt_list, pred_list)
        ap_list.append(ap)
    print(ap_list)
    mAP = np.mean(ap_list)

    return mAP


# 从label.json和pred.json加载数据
with open("label.json", "r") as f:
    label_data = json.load(f)

with open("pred.json", "r") as f:
    pred_data = json.load(f)

# 根据Image_Name排序数据
sorted_label = sorted(label_data, key=lambda x: x["Image_Name"])
sorted_pred = sorted(pred_data, key=lambda x: x["Image_Name"])


mAP = calculate(sorted_label, sorted_pred)
print(f"mAP: {mAP}")

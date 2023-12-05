import json
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def calculate_ap(gt, pred):
    precision, recall, _ = precision_recall_curve(gt, pred)
    ap = auc(recall, precision)
    return ap


def calculate(label_data, pred_data):
    ap_list = []

    for _, class_name in enumerate(classes):
        gt_list = [sample[class_name] for sample in label_data]
        pred_list = [sample[class_name] for sample in pred_data]

        ap = calculate_ap(gt_list, pred_list)
        ap_list.append(ap)

    mAP = np.mean(ap_list)

    return mAP


# 从label.json和pred.json加载数据
with open("train.json", "r") as f:
    label_data = json.load(f)

with open("pred.json", "r") as f:
    pred_data = json.load(f)

# 根据Image_Name排序数据
sorted_label = sorted(label_data, key=lambda x: x["name"])
sorted_pred = sorted(pred_data, key=lambda x: x["name"])

mAP = calculate(sorted_label, sorted_pred)

print(f"mAP: {mAP}")

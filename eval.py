import json

def calculate_map(label_data, pred_data):
    precision = 0.
    recall = 0.
    acc = 0.
    classes = ['motorcycle', 'truck', 'boat', 'bus', 'cycle', 'sitar', 'ektara', 'flutes', 'tabla', 'harmonium']

    for class_name in classes:
        tp = 0.
        fp = 0.
        tn = 0.
        fn = 0.
        num_samples = len(label_data)

        for i in range(num_samples):
            
            gt = label_data[i][class_name]
            pred = pred_data[i][class_name]

            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 1 and pred == 0:
                fn += 1
            elif gt == 0 and pred == 0:
                tn += 1

        precision += tp / (tp + fp) if (tp + fp) > 0 else 0

        recall += tp / (tp + fn) if (tp + fn) > 0 else 0

        acc += (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

    mAP = precision / len(classes)
    mAR = recall / len(classes)
    mACC = acc / len(classes)

    return mAP, mAR, mACC

# 从label.json和pred.json加载数据
with open('label.json', 'r') as f:
    label_data = json.load(f)

with open('pred.json', 'r') as f:
    pred_data = json.load(f)

sorted_data = sorted(pred_data, key=lambda x: x["Image_Name"])

print(label_data[0]['motorcycle'])

mAP, mAR, mACC = calculate_map(label_data, sorted_data)

print(f'mAP: {mAP}')
print(f'mAR: {mAR}')
print(f'mACC: {mACC}')


import cv2
import torch
import yaml
from tqdm import tqdm
import os
import time
import json
from sklearn.metrics import average_precision_score
import numpy as np
from torch import Tensor

import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

def asign_label_id(text1):
    """Check if two texts are similar."""
    if 'DITY' in text1.upper():
        return 5    
    elif 'CY' in text1.upper():
        if 'J' in text1.upper():
            return 4
        else:
            return 1
    elif 'PRAT' in text1.upper():
        return 3
    elif 'GL' in text1.upper():
        return 2
    elif 'ME' in text1.upper():
        return 0

    return -1

def load_dataset_yaml(dataset_yaml_path: str = "coco8.yaml") -> list:
    """Load categories names from a dataset YAML file in YOLO farmat."""
    with open(dataset_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

def compute_iou(pred_box1, anno_box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = pred_box1
    x1g, y1g, x2g, y2g = anno_box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def load_image_and_labels(image_path: str, label_path: str):
    """Load an image and its corresponding labels."""
    image = Image.open(image_path).convert("RGB")

    pixel_width, pixel_height = image.size

    with open(label_path, 'r') as file:
        raw_labels = [line.strip().split() for line in file.readlines()]

        labels = []

        for raw_label in raw_labels:
            if len(raw_label) == 5:
                category, cx, cy, w, h = raw_label
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)

                labels.append({
                    'category_id': int(category),
                    'bbox': [(cx - 0.5*w) * pixel_width, (cy - 0.5*h) * pixel_height, (cx + 0.5*w) * pixel_width, (cy + 0.5*h) * pixel_height]
                    })


    return image, labels

def model_validation(
    data: dict,
    model: torch.nn.Module,
    save_path: str = None):

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", padding=True, truncation=True)

    if save_path is None:
        save_path = "results"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    categories = []

    for i, category in data.get("names").items():
        categories.append(category)

    image_dir = data['val']
    label_dir = image_dir.replace('images', 'labels')

    image_files = [f for f in os.listdir(os.path.join(data['path'], image_dir)) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG') or f.endswith('.jpeg') or f.endswith('.JPEG')]

    all_results = []
    all_labels = []
    all_scores = []
    all_ious = []

    t = 0
    for image_file in tqdm(image_files):
        image_path = os.path.join(data['path'], image_dir, image_file)
        file_name = ".".join(image_file.split(".")[:-1])
        label_path = os.path.join(data['path'], label_dir, f"{file_name}.txt")

        image, annos = load_image_and_labels(image_path, label_path)
        
        inputs = processor(text=[categories], images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items() if isinstance(v, Tensor) }
        
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()
        t += (end_time - start_time)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.01, target_sizes=target_sizes)


        boxes, scores, labels = results[0]["boxes"].cpu().detach().numpy().tolist(), results[0]["scores"].cpu().detach().numpy().tolist(), results[0]["labels"].cpu().detach().numpy().tolist()

        for box, score, label in zip(boxes, scores, labels):
            all_results.append({
                    'image_id': image_file,
                    'category_id': categories[label],
                    'bbox': box,
                    'score': score
                    })
            all_scores.append(score)
            all_labels.append(categories[label])
            ious = [compute_iou(box, l['bbox']) for l in annos if l['category_id'] == label]

            all_ious.append(max(ious) if ious else 0)

            if len(ious) > 1:
                print(f"Multiple ious: {ious}")

        # Save results to file
        with open(os.path.join(save_path, 'results.json'), 'w') as f:
            json.dump(all_results, f)

    detection_speed = t / len(image_files)

    # Print detection speed
    print(f"Detection speed: {detection_speed} seconds per image")

    iou_thresholds = list(np.arange(0.5, 0.95, 0.05))

    # Print mAP with different I
    aps = []
    for iou_thresh in iou_thresholds:
        y_true = [1 if iou >= iou_thresh else 0 for iou in all_ious]
        ap = average_precision_score(y_true, all_scores)
        print(f"AP at IoU {iou_thresh}: {ap}")
        aps.append(ap)

    print(f"mAP: {np.mean(np.array(aps))}")

    # Print mAP for each class with IoU 0.5
    aps_per_class = []
    for i, category in enumerate(categories):
        y_true = [1 if iou >= 0.5 and label == category else 0 for iou, label in zip(all_ious, all_labels)]
        if sum(y_true) == 0:
            ap = 0
        else:
            ap = average_precision_score(y_true, all_scores)
        print(f"AP for class {category}: {ap}")
        aps_per_class.append(ap)

    mAP = sum(aps_per_class) / len(aps_per_class)
    AP_s = sum(aps_per_class[:11]) / 11
    AP_u = sum(aps_per_class[11:]) / 16
    AP_h = 2 * AP_s * AP_u / (AP_s + AP_u)

    print(f"mAP: {mAP}")
    print(f"AP_s: {AP_s}")
    print(f"AP_u: {AP_u}")
    print(f"AP_h: {AP_h}")

    results_json = {
        "mAP": mAP,
        "AP_s": AP_s,
        "AP_u": AP_u,
        "AP_h": AP_h,
        "FPS": 1/detection_speed
    }

    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results_json, f)

if __name__ == "__main__":

    data_yaml_file = "uk_pest_dataset_07JAN25_all_insect.yaml"
    # Load the dataset YAML file
    data = load_dataset_yaml(data_yaml_file)

    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").cuda()

    model_validation(data, model)
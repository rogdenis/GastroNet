import random
import json
import sys
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import os
import base64
import requests
import albumentations as A
from requests.adapters import HTTPAdapter, Retry
from time import sleep
from copy import copy
from pprint import pprint
from pycocotools.coco import COCO
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import ClassificationDataset, collate_fn
from torchmetrics import PrecisionRecallCurve
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from torchvision.models import resnet50

TYPES = {
    "SEGMENTATION": {
        "KEY": "dKmUZU6MLwX9R504E4M1",
        "PREFIX": "https://outline.roboflow.com/instance-d8qr5/1"
    },
    "CLASSIFICATION": {
        "KEY": "YQYAOWDsYtkXsusKdvxd",
        "PREFIX": "https://classify.roboflow.com/endo-navi-gastro-v-1.0/1"
    }
}

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[104, 500, 502, 503, 504] )
session.mount('https://', HTTPAdapter(max_retries=retries))

image_transform = A.Compose([
    A.MotionBlur(p=1),
    A.Defocus(p=1)
])

coords_transform = A.Compose([
    A.Affine(p=1),
    A.Flip(p=1)
])

test = ClassificationDataset('classification/test', '_classes.csv')

classes = ('Antrum pyloricum', 'Antrum pyloricun', 'Corpus gastricum', 'Duodenum', 'Esophagus III/III', 'Mouth', 'Oesophagus', 'Oropharynx', 'Pars cardiaca')

test_dataloader = DataLoader(test, batch_size=1, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = resnet50()
PATH = "classification.pt"
if len(sys.argv) > 1: PATH = sys.argv[1]
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("epoch", checkpoint['epoch'])
writer = SummaryWriter('classification')

i = 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
total_roboflow_pred = {classname: 0 for classname in classes}
total_label = {classname: 0 for classname in classes}
correct = 0
roboflow = 0
total = 0
for images, labels in test_dataloader:
    #metric = MeanAveragePrecision(class_metrics=True, iou_thresholds = [0.9], rec_thresholds=[0.001])
    frames = tuple(image.numpy().astype(np.uint8).swapaxes(0, 2).swapaxes(0, 1) for image in images)#.swapaxes(0, 2).swapaxes(1, 0) for image in images)
    #images = list(image.to(device) for image in images)
    with torch.no_grad():
        outputs = model(images.to(device))
    _, predictions = torch.max(outputs, 1)
    img_name = "frame" + str(i)
    retval, buffer = cv2.imencode('.jpg', frames[0])
    jpg_as_text = base64.b64encode(buffer)
    segmentation_url = "{}?api_key={}&name={}.jpg".format(
        TYPES['CLASSIFICATION']["PREFIX"],
        TYPES['CLASSIFICATION']["KEY"],
        img_name)
    r = session.post(segmentation_url, data=jpg_as_text, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })
    r_predictions = r.json()['predictions']
    roboflow_prediction = max(r_predictions, key=lambda k: r_predictions[k]['confidence'])
    for label, prediction in zip(labels, predictions):
        if prediction == label:
            correct_pred[classes[label]] += 1
            correct += 1
        if roboflow_prediction == classes[label]:
            roboflow += 1
        total += 1
        total_pred[classes[prediction]] += 1
        total_label[classes[label]] += 1
        text = classes[prediction]
        COLOR = (0, 128, 0) if prediction == label else (255, 0, 0)
        image = cv2.putText(frames[0].copy(), text, (30,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
        cv2.imwrite("test_classification/image{}.jpg".format(i), image)
        print("image", i, classes[label], text, prediction == label, roboflow_prediction, roboflow_prediction == classes[label])
    i += 1

accuracy = 100 * correct / total
print(f'My accuracy: {accuracy} %')

roboflow_accuracy = 100 * roboflow / total
print(f'Roboflow accuracy: {roboflow_accuracy} %')

for classname, correct_count in correct_pred.items():
    precision = 100 * float(correct_count) / total_pred[classname] if total_pred[classname] > 0 else 0
    recall = 100 * float(correct_count) / total_label[classname] if total_label[classname] > 0 else 0
    print(f'{classname:5s} Precision: {precision:.1f}, Recall: {recall:.1f}, Count Pred: {total_pred[classname]}, Count GT: {total_label[classname]}')

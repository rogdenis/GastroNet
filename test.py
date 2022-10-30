import random
import json
import sys
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import os
import albumentations as A
from time import sleep
from copy import copy
from pprint import pprint
from pycocotools.coco import COCO
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import filter_by_threshold, draw_segmentation_map, filter_nms, calculate_metrics, convert_to_array
from utils import CustomImageDataset, collate_fn
from torchmetrics import PrecisionRecallCurve
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
torch.manual_seed(0)
TH = 0

image_transform = A.Compose([
    A.MotionBlur(p=1),
    A.Defocus(p=1)
])

coords_transform = A.Compose([
    A.Affine(p=1),
    A.Flip(p=1)
])

train = CustomImageDataset('Gastro.v1i.coco-segmentation/train', '_annotations.coco.json')
test = CustomImageDataset('Gastro.v1i.coco-segmentation/test', '_annotations.coco.json',
    # image_transform=image_transform,
    # coords_transform=coords_transform,
    cats=train.cats,
    bg=False)
print(train.cats)

train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=6)  # replace the pre-trained head with a new one
PATH = "last.pt"
if len(sys.argv) > 1: PATH = sys.argv[1]
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)# move model to the right devic
model.eval()
print("epoch", checkpoint['epoch'])
print("params", checkpoint['params'])
writer = SummaryWriter(checkpoint['params'])

i = 0
GL = {str(th/100): {"TP":0, "FP":0, "FN":0} for th in range(0,100,5)}
for images, targets in test_dataloader:
    #metric = MeanAveragePrecision(class_metrics=True, iou_thresholds = [0.9], rec_thresholds=[0.001])
    frames = tuple(image.numpy().astype(np.uint8).swapaxes(0, 2).swapaxes(0, 1) for image in images)#.swapaxes(0, 2).swapaxes(1, 0) for image in images)
    images = list(image.to(device) for image in images)
    with torch.no_grad():
        outputs = model(images)
    outputs = filter_nms(outputs)
    filtered_outs = filter_by_threshold(outputs, TH)
    predict = draw_segmentation_map(frames[0].copy(), filtered_outs[0])
    gt = draw_segmentation_map(frames[0].copy(), targets[0])
    image = np.hstack([gt,predict])
    preds, gt = convert_to_array(filtered_outs[0], targets[0])
    TP, FP, FN = calculate_metrics(preds, gt, 6, 0.5)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    text = "PR: {:.2f}, RE: {:.2f}".format(precision, recall)
    text_pos = (image.shape[1] // 2 - 150, image.shape[0] - 50)
    image = cv2.putText(image.copy(), text, (text_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("test_detections/image{}.jpg".format(i), image)
    print("image", i, text)

    for th, vals in GL.items():
        filtered_outs = filter_by_threshold(outputs, float(th))
        preds, gt = convert_to_array(filtered_outs[0], targets[0])
        #print(gt)
        TP, FP, FN = calculate_metrics(preds, gt, 6, 0.5)
        vals["TP"] += TP
        vals["FP"] += FP
        vals["FN"] += FN
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    i += 1

i = 0
for th, vals in GL.items():
    if (vals["TP"] + vals["FP"]) > 0 and (vals["TP"] + vals["FN"]) > 0:
        precision = vals["TP"] / (vals["TP"] + vals["FP"])
        recall = vals["TP"] / (vals["TP"] + vals["FN"])
        writer.add_scalars('test-thersholds_{}'.format(checkpoint['params'].replace("/","")), {
            "precision": precision,
            "recall": recall,
            "falses": vals["FP"] / len(test_dataloader)},
            int(float(th) * 100))
        sleep(0.1)
        print(int(float(th) * 100))
        print("TH: {} PR: {:.2f}, RE: {:.2f}, FR: {:.2f}".format(
            th, precision, recall, vals["FP"] / len(test_dataloader)))
        i += 1
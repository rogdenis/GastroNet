import random
import json
import sys
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import os
from copy import copy
from pprint import pprint
from pycocotools.coco import COCO
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import filter_by_threshold, draw_segmentation_map, filter_nms, calculate_metrics, convert_to_array
from torchmetrics import PrecisionRecallCurve
from pprint import pprint
TH = 0.85


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file, cats=None, transform=None, target_transform=None, bg=True):
        self.img_dir = img_dir
        self.coco = COCO(os.path.join(img_dir, annotations_file))
        if cats is None:
            self.cats = {v: k+1 for k, v in dict(enumerate(self.coco.getCatIds())).items()}
        else:
            self.cats = cats
        self.transform = transform
        self.target_transform = target_transform
        self.bg = bg

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.coco.imgs[idx]['file_name'])
        image = read_image(img_path)
        image = torch.as_tensor(image, dtype=torch.float32)
        #image = F.convert_image_dtype(image)
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        num_objs = len(ann_ids)
        masks = []
        labels = []
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        if self.bg:
            num_objs += 1
            boxes = torch.zeros([num_objs,4], dtype=torch.float32)
            bg = np.zeros(image.size()[1:],dtype=np.uint8)
            masks = [bg]
            labels = [0]
            #boxes[0] = torch.tensor([0, 0, image.size()[1], image.size()[2]])
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)
            mask = self.coco.annToMask(ann[0])
            rows, cols = np.nonzero(mask)
            masks.append(mask)
            labels.append(self.cats[ann[0]['category_id']])
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] = torch.tensor(labels, dtype=torch.int64)#torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, data

def collate_fn(batch):
    return tuple(zip(*batch))

train = CustomImageDataset('Gastro.v1i.coco-segmentation/train', '_annotations.coco.json')
test = CustomImageDataset('Gastro.v1i.coco-segmentation/test', '_annotations.coco.json', cats=train.cats, bg=False)
print(train.cats)

train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=6)  # replace the pre-trained head with a new one
PATH = "last.pt"
if len(sys.argv) > 1: PATH = sys.argv[1]
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)# move model to the right devic
model.eval()
print(checkpoint['epoch'])

i = 0
GL = {str(th/100): {"TP":0, "FP":0, "FN":0} for th in range(0,100,5)}
for images, targets in test_dataloader:
    #metric = MeanAveragePrecision(class_metrics=True, iou_thresholds = [0.9], rec_thresholds=[0.001])
    frames = tuple(image.numpy().astype(np.uint8).swapaxes(0, 2).swapaxes(0, 1) for image in images)
    images = list(image.to(device) for image in images)
    with torch.no_grad():
        outputs = model(images)
    outputs = filter_nms(outputs)
    filtered_outs = filter_by_threshold(outputs, TH)
    predict = draw_segmentation_map(frames[0], filtered_outs[0])
    gt = draw_segmentation_map(frames[0], targets[0])
    image = np.hstack([gt,predict])
    filtered_outs = filter_by_threshold(outputs, TH)
    gt, preds = convert_to_array(filtered_outs[0], targets[0])
    TP, FP, FN = calculate_metrics(gt, preds, 6, 0.5)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    for th, vals in GL.items():
        TH = float(th)
        filtered_outs = filter_by_threshold(outputs, TH)
        preds, gt = convert_to_array(filtered_outs[0], targets[0])
        #print(gt)
        TP, FP, FN = calculate_metrics(preds, gt, 6, 0.5)
        vals["TP"] += TP
        vals["FP"] += FP
        vals["FN"] += FN
    print(i)
    #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    

    text = "PR: {:.2f}, RE: {:.2f}".format(precision, recall)
    image = cv2.putText(image, text, (image.shape[1] // 2 - 150, image.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("test_detections/image{}.jpg".format(i), image)
    i += 1


for th, vals in GL.items():
    precision = vals["TP"] / (vals["TP"] + vals["FP"])
    recall = vals["TP"] / (vals["TP"] + vals["FN"])
    print("TH: {} PR: {:.2f}, RE: {:.2f}, FR: {:.2f}".format(
        th, precision, recall, vals["FP"] / len(test_dataloader)))
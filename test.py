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
from utils import draw_segmentation_map, get_outputs, loadData, filter_by_threshold, draw_segmentation_map
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import PrecisionRecallCurve
from pprint import pprint
TH = 0.8


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file, cats=None, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.coco = COCO(os.path.join(img_dir, annotations_file))
        if cats is None:
            self.cats = {v: k+1 for k, v in dict(enumerate(self.coco.getCatIds())).items()}
        else:
            self.cats = cats
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.coco.imgs[idx]['file_name'])
        image = read_image(img_path)
        image = torch.as_tensor(image, dtype=torch.float32)
        #image = F.convert_image_dtype(image)
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        bg = np.zeros(image.size()[1:],dtype=np.uint8)
        masks = [bg]
        labels=[0]
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)
            mask = self.coco.annToMask(ann[0])
            rows, cols = np.nonzero(mask)
            # print(mask[rows, cols])
            # print(self.cats[ann[0]['category_id']])
            masks.append(mask)
            labels.append(self.cats[ann[0]['category_id']])
        num_objs = len(masks)
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        boxes[0] = torch.tensor([0, 0, image.size()[1], image.size()[2]])
        for i in range(1,num_objs):
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

def drawmasks(frame, model, device):
    orig_image = frame.copy()
    image = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
    image = image.swapaxes(1, 3).swapaxes(2, 3)
    masks, boxes, labels = get_outputs(image.to(device), model, TH)
    result = draw_segmentation_map(orig_image, masks, boxes, labels)
    #print(result.shape)
    #cv2.imwrite("detection1.png", result)
    return result, len(boxes)

train = CustomImageDataset('Gastro.v1i.coco-segmentation/train', '_annotations.coco.json')
test = CustomImageDataset('Gastro.v1i.coco-segmentation/test', '_annotations.coco.json', cats=train.cats)
print(train.cats)

train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=6)  # replace the pre-trained head with a new one
checkpoint = torch.load("best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)# move model to the right devic
model.eval()
print(checkpoint['epoch'])

i = 0
main_metric = {str(th/10):0 for th in range(11)}
for images, targets in test_dataloader:
    frame = images[0].cpu().detach().numpy()
    img_copy = copy(frame)
    # images = list(image.to(device) for image in images)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    # outputs = model(images)
    # predict = draw_segmentation_map(frame, outputs[0]["masks"], outputs[0]["boxes"], outputs[0]["labels"])
    s = cv2.imread("image.jpg")
    # predict = predict.swapaxes(0, 2)
    # print(predict.shape)
    # #print(predict)
    # print(s.shape)
    # #print(s)
    frame = frame.astype(int).swapaxes(0, 2).swapaxes(0, 1)
    print(np.histogram(frame.flatten()))
    print(np.histogram(s.flatten()))
    print(cv2.imwrite("predict.jpg", img_copy*255))
    #print(frame)
    #gt = draw_segmentation_map(frame, targets[0]["masks"], targets[0]["boxes"], targets[0]["labels"])
    #cv2.imwrite("gt.jpg",gt)
    print("done")
    break
    
#     for score in main_metric.keys():
#         metric = MeanAveragePrecision(class_metrics=True, rec_thresholds=[0.8], iou_thresholds = [0.5])
#         metric.update(filter_by_threshold(outputs, float(score)), targets)
#         main_metric[score] += torch.tensor(metric.compute()["map"]).item()
#     i += 1

# for score, value in main_metric.items():
#     print("{}: {}".format(score, value / len(test_dataloader)))

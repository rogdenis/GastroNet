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
from utils import draw_segmentation_map, get_outputs, loadData
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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


train = CustomImageDataset('Gastro.v1i.coco-segmentation/train', '_annotations.coco.json')
valid = CustomImageDataset('Gastro.v1i.coco-segmentation/valid', '_annotations.coco.json', cats=train.cats)
print(train.cats)

train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid, batch_size=1, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(train.cats))  # replace the pre-trained head with a new one
model.load_state_dict(torch.load("epoch_39.torch"))
model.to(device)# move model to the right devic
model.eval()

metrics = {'count': 0}
i = 0
for images, targets in valid_dataloader:
    if targets[0]["boxes"].size()[0] == 1:
        continue
    outputs = model(images)
    metric = MeanAveragePrecision()
    metric.update(outputs, targets)
    i += 1
    # locs = copy(locals())
    # for l, v in {k: v for k, v in sorted(locs.items(), key=lambda item: -sys.getsizeof(item[0]))}.items():
    #     print(sys.getsizeof(locals()[l]), l)
    if i > 30:
        break
        
pprint(metric.compute())
#DONE #https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
#DONE #https://zenodo.org/record/4736111/files/LabPicsChemistry.zip?download=1
#DONE #https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code
#DONE валидация
#https://towardsdatascience.com/a-comprehensive-guide-to-image-augmentation-using-pytorch-fb162f2444be - аугментация
#GPU

import random
import json
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
from pycocotools.coco import COCO

batchSize=2
imageSize=[800,600]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
#trainDir="/home/rogdenis/segmentation/LabPicsChemistry/Train"
trainDir="/home/rogdenis/segmentation/Gastro.v1i.coco-segmentation/train"

imgs=[]
for pth in os.listdir(trainDir):
    imgs.append(trainDir+"/"+pth +"/")

coco = COCO('Gastro.v1i.coco-segmentation/train/_annotations.coco.json')
imgIds = coco.getImgIds()
cats = {v: k for k, v in dict(enumerate(coco.getCatIds())).items()}
print(cats)
    
UNIQUE_LABELS = {
    101:1,
    110:2,
    118:3,
    119:4,
    121:5,
    122:6,
    124:7
}

def loadData():
    batch_Imgs=[]
    batch_Data=[]# load images and masks
    for i in range(batchSize):
        idx=random.randint(0,len(imgIds)-1)
        file_name = coco.imgs[idx]['file_name']
        img = cv2.imread(os.path.join(trainDir, file_name))
        print(img)
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        ann_ids = coco.getAnnIds(imgIds=idx)
        masks=[]
        labels=[]
        for ann_id in ann_ids:
            ann = coco.loadAnns(ann_id)
            mask = coco.annToMask(ann[0])
            mask=cv2.resize(mask,imageSize,cv2.INTER_NEAREST)
            masks.append(mask)
            labels.append(cats[ann[0]['category_id']])
        num_objs = len(masks)
        if num_objs==0: return loadData() # if image have no objects just load another image
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] = torch.tensor(labels, dtype=torch.int64)#torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    print(batch_Imgs.shape)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    print(batch_Imgs.shape)
    return batch_Imgs, batch_Data


def loadData_():
    batch_Imgs=[]
    batch_Data=[]# load images and masks
    for i in range(batchSize):
        idx=random.randint(0,len(imgs)-1)
        img = cv2.imread(os.path.join(imgs[idx], "Image.jpg"))
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        maskDir=os.path.join(imgs[idx], "Vessels")
        masks=[]
        labels=[]
        with open(os.path.join(imgs[idx], "Data.json")) as lf:
            labeldata = json.loads(lf.read())
        #for mskName in os.listdir(maskDir):
        for k, vessel in labeldata["Vessels"].items():
            vessel_type = UNIQUE_LABELS.get(vessel["VesselType_ClassIDs"][-1], 0)
            labels.append(vessel_type)
            MaskFilePath = vessel["MaskFilePath"]
            vesMask = (cv2.imread(imgs[idx]+MaskFilePath, 0) > 0).astype(np.uint8)  # Read vesse instance mask
            vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST)
            #print(vesMask)
            masks.append(vesMask)# get bounding box coordinates for each mask
        num_objs = len(masks)
        if num_objs==0: return loadData() # if image have no objects just load another image
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] = torch.tensor(labels, dtype=torch.int64)#torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=8)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

for i in range(10001):
    images, targets = loadData()
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad()
    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()
    print(i,'loss:', losses.item())
    if i%500==0:
        torch.save(model.state_dict(), str(i)+".torch")
#DONE #https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
#DONE #https://zenodo.org/record/4736111/files/LabPicsChemistry.zip?download=1
#DONE #https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code
#DONE валидация
#метрики
#https://towardsdatascience.com/a-comprehensive-guide-to-image-augmentation-using-pytorch-fb162f2444be - аугментация
#GPU
#тензорборда

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
from utils import loadData, make_batch

batchSize=2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available

dataset, cats = loadData("/home/rogdenis/GastroNet/Gastro.v1i.coco-segmentation/train", augmentation=True)
print(len(dataset))
#print(dataset[0][1])
#train_data = make_batches(dataset, batchSize)
val_dataset, cats_val = loadData("/home/rogdenis/GastroNet/Gastro.v1i.coco-segmentation/valid", cats=cats)
#val_data = make_batches(val_dataset, 1)

#print(len(train_data))
num_classes = len(cats.keys())
print(num_classes)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
model.train()
for epoch in range(100):
    print("Train epoch {}".format(epoch))
    start = 0
    while len(dataset) - start >= batchSize:
        images, targets, start = make_batch(dataset, batchSize, start)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(start - 1,'loss:', losses.item())
    
    val_losses = []
    start = 0
    while len(val_dataset) - start >= 1:
        images, targets, start = make_batch(dataset, 1, start)
        val_dict = model(images, targets)
        val_losses.append(float(sum(loss.detach() for loss in val_dict.values())))
    print('{} epoch loss: {}'.format(epoch,sum(val_losses)/len(val_losses)))

    lr_scheduler.step()
    torch.save(model.state_dict(), "epoch_{}.torch".format(epoch))
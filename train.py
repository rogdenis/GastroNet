#DONE #https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
#DONE #https://zenodo.org/record/4736111/files/LabPicsChemistry.zip?download=1
#DONE #https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code
#DONE валидация
#DONE метрики
#DONE https://towardsdatascience.com/a-comprehensive-guide-to-image-augmentation-using-pytorch-fb162f2444be - аугментация
#DONE GPU
#DONE тензорборда
#DONE https://pytorch.org/tutorials/beginner/saving_loading_models.html - загрузка моделей
#DONE https://stackoverflow.com/questions/67295494/correct-validation-loss-in-pytorch criterion

import random
import json
import numpy as np
import cv2
import os
import torch
import sys
import torchvision.models.segmentation
import albumentations as A
from torchvision import transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from utils import filter_by_threshold, filter_nms, calculate_metrics, convert_to_array, CustomImageDataset, collate_fn
from mean_average_precision import MetricBuilder

batchSize=2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
writer = SummaryWriter('runs/train_segmentation')

image_transform = A.Compose([
    A.MotionBlur(p=0.5),
    A.Defocus(p=0.5)
])

coords_transform = A.Compose([
    A.Affine(p=0.5),
    A.Flip(p=0.5)
])

train = CustomImageDataset('Gastro.v1i.coco-segmentation/train', '_annotations.coco.json',
    image_transform=image_transform,
    coords_transform=coords_transform,
    bg=True)
valid = CustomImageDataset('Gastro.v1i.coco-segmentation/valid', '_annotations.coco.json', cats=train.cats, bg=True)
print(train.__len__())
print(train.cats)

train_dataloader = DataLoader(train, batch_size=8,
    shuffle=True,
    collate_fn=collate_fn)
valid_dataloader = DataLoader(valid, batch_size=8, shuffle=True, collate_fn=collate_fn)

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=1+len(train.cats))  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.1,
                                                verbose=True)
start_epoch = 0
last_best = -1
if len(sys.argv) > 1 and sys.argv[1] == 'continue':
    if len(sys.argv) > 2:
        PATH = sys.argv[2]
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    last_best = checkpoint['metric']

for epoch in range(start_epoch,100):
    print("Train epoch {}".format(epoch))
    model.train()
    i = 0
    for images, targets in train_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        writer.add_scalars('training-loss_parts', loss_dict, epoch * len(train_dataloader) + i)
        writer.add_scalar('training-loss', losses, epoch * len(train_dataloader) + i)
        losses.backward()
        optimizer.step()
        i += 1
        # if i % 20 == 0:
        #     print(i,'loss:', losses.item())

    #VALIDATE
    #model.eval()
    main_metric = 0
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=6)
    losses = 0
    with torch.no_grad():
        for images, targets in valid_dataloader:
            model.train()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses += sum(loss for loss in loss_dict.values())
            model.eval()
            outputs = model(images)
            outputs = filter_nms(outputs)
            filtered_outs = filter_by_threshold(outputs, 0)
            for out, target in zip(filtered_outs, targets):
                preds, gt = convert_to_array(out, target)
                metric_fn.add(preds, gt)

    main_metric = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

    print('epoch {} map: {}'.format(epoch, main_metric))
    
    writer.add_scalars('epoch-map', {"mAP": main_metric, "loss": losses / len(valid_dataloader)}, epoch)
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            'metric': main_metric
            }
    torch.save(checkpoint, "last.pt")

    if main_metric > last_best:
        last_best = main_metric
        torch.save(checkpoint, "best_{}.pt".format(optimizer.param_groups[0]['lr']))

    lr_scheduler.step()


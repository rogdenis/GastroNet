#DONE #https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
#DONE #https://zenodo.org/record/4736111/files/LabPicsChemistry.zip?download=1
#DONE #https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code
#DONE валидация
#DONE метрики
#https://towardsdatascience.com/a-comprehensive-guide-to-image-augmentation-using-pytorch-fb162f2444be - аугментация
#GPU
#DONE тензорборда
#DONE https://pytorch.org/tutorials/beginner/saving_loading_models.html - загрузка моделей

import random
import json
import numpy as np
import cv2
import os
import torch
import sys
import torchvision.models.segmentation
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from utils import filter_by_threshold, filter_nms, calculate_metrics, convert_to_array
from mean_average_precision import MetricBuilder

batchSize=2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
writer = SummaryWriter('runs/train_segmentation')

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
            boxes[0] = torch.tensor([0, 0, image.size()[2], image.size()[1]])
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)
            mask = self.coco.annToMask(ann[0])
            rows, cols = np.nonzero(mask)
            masks.append(mask)
            labels.append(self.cats[ann[0]['category_id']])
        for i in range(1, num_objs):
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
valid = CustomImageDataset('Gastro.v1i.coco-segmentation/valid', '_annotations.coco.json', cats=train.cats, bg=False)
print(train.__len__())
print(train.cats)

train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=1+len(train.cats))  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
start_epoch = 0
last_best = -1
if len(sys.argv) > 1 and sys.argv[1] == 'continue':
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
        writer.add_scalar('training-loss', losses, epoch * len(train_dataloader) + i)
        losses.backward()
        optimizer.step()
        i += 1
        if i % 20 == 0:
            print(i,'loss:', losses.item())

    val_losses = []
    start = 0
    model.eval()
    main_metric = 0
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=6)
    for images, targets in valid_dataloader:
        images = list(image.to(device) for image in images)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs = model(images)
        outputs = filter_nms(outputs)
        filtered_outs = filter_by_threshold(outputs, 0)
        preds, gt = convert_to_array(filtered_outs[0], targets[0])
        metric_fn.add(preds, gt)

    main_metric = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

    print('epoch {} map: {}'.format(epoch, main_metric))
    
    writer.add_scalar('epoch-map', main_metric, epoch)
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
        torch.save(checkpoint, "best.pt")

    lr_scheduler.step()


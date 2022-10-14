import random
import json
import sys
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
from utils import draw_segmentation_map, get_outputs, loadData
imgPath=sys.argv[1]
TH = 0.8

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=5)  # replace the pre-trained head with a new one
model.load_state_dict(torch.load("epoch_39.torch"))
model.to(device)# move model to the right devic
model.eval()

image=cv2.imread(imgPath)
print(image.shape)
image = cv2.resize(image, imageSize, cv2.INTER_LINEAR)
orig_image = image.copy()
image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
image=image.swapaxes(1, 3).swapaxes(2, 3)
masks, boxes, labels = get_outputs(image, model, TH)
result = draw_segmentation_map(orig_image, masks, boxes, labels)
cv2.imwrite("detection.png", np.hstack([orig_image,result]))

# train_data, cats = loadData("/home/rogdenis/segmentation/Gastro.v1i.coco-segmentation/train", 1, onlycats=True)
# test_data, test_val = loadData("/home/rogdenis/segmentation/Gastro.v1i.coco-segmentation/test", 1, cats=cats)

# losses = []
# for images, targets in test_data:
#     test_dict = model(images, targets)
#     losses.append(float(sum(loss.detach() for loss in test_dict.values())))
#     # masks, boxes, labels = get_outputs(images[0], model, TH)
#     # result = draw_segmentation_map(orig_image, masks, boxes, labels)
#     # cv2.imwrite("detection.png", np.hstack([orig_image,result]))
    
# print('{} epoch loss: {}'.format(epoch,sum(val_losses)/len(val_losses)))
#https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/

import torch
import torchvision
import cv2
import argparse
import sys
from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                           num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

#!c1.8
image_path = sys.argv[1]
TH = 0.5
size = 1200, 800

image = Image.open(image_path).convert('RGB')
# keep a copy of the original image for OpenCV functions and applying masks

# transform the image
image = image.resize(size)
orig_image = image.copy()
image.save("resized.jpg")
image = transform(image)
# add a batch dimension
image = image.unsqueeze(0).to(device)
masks, boxes, labels = get_outputs(image, model, TH)
print((len(masks), len(labels), len(boxes)))
result = draw_segmentation_map(orig_image, masks, boxes, labels)
# visualize the image
#cv2.imshow('Segmented image', result)
#cv2.waitKey(0)
# set the save path
save_path = "detections_"+sys.argv[1]
cv2.imwrite(save_path, result)
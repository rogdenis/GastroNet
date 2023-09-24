import requests
import sys
import cv2
import base64
import io
import json
import torch
import random
import boto3
from time import time
import numpy as np
import os.path
import urllib.parse
import torchvision.models.segmentation
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import draw_segmentation_map, filter_by_threshold, filter_nms, get_colors, SegmentationDataset
from torchvision.models import resnet50

TH = float(sys.argv[1])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available


train = SegmentationDataset('dataset20230117', 'annotations_coco.json')
print(train.cats)

#LOAD SEGMENTATION MODEL
PATH = "best_segmentation.pt"
if len(sys.argv) > 2: PATH = sys.argv[2]
checkpoint = torch.load(PATH)
segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = segmentation_model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
segmentation_model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=1+len(train.cats))  # replace the pre-trained head with a new one
segmentation_model.load_state_dict(checkpoint['model_state_dict'])
segmentation_model.to(device)# move model to the right devic
segmentation_model.eval()

classes = [
    'Antrum pyloricum',
    'Corpus gastricum',
    'Duodenum',
    'Esophagus',
    'Mouth',
    'Oropharynx',
    'Void']

COLORS, COCO_NAMES = get_colors('dataset20230117', 'annotations_coco.json')

#LOAD CLASSIFICATION MODEL
PATH = "classification.pt"
if len(sys.argv) > 3: PATH = sys.argv[3]
checkpoint = torch.load(PATH)
classification_model = resnet50(num_classes=1000)#len(classes))
classification_model.load_state_dict(checkpoint['model_state_dict'])
classification_model.to(device)# move model to the right devic
classification_model.eval()


SCORES = [0] * len(classes)
A = 0.2


def getVideostat(URL, segmentation_model, classification_model):
    PATHOLOGIES = {pathologie: 0 for pathologie in COCO_NAMES}
    CLASSES = {part: 0 for part in classes}
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1)
    session.mount('http://', HTTPAdapter(max_retries=retries))
    softmax = torch.nn.Softmax(dim=0)
    video_in = cv2.VideoCapture(URL)
    ok, frame = video_in.read()
    orig_frame = frame.copy()
    height,width,layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #video_out=cv2.VideoWriter('detections_' + VIDEO_NAME, fourcc, 20.0, (2 * width,height))
    scores = []
    switches = 0
    pred = None
    i = 0
    while ok:
        image = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
        image = image.swapaxes(1, 3).swapaxes(2, 3).to(device)
        with torch.no_grad():
            detections = segmentation_model(image)
            detections = filter_nms(detections)
            detections = filter_by_threshold(detections, TH)
            classification = classification_model(image)
            classification = softmax(classification_model(image)[0][:len(classes)])
            classification = classification.cpu().numpy().tolist()
        ps = max(classification)
        NAVIGATION_CLASS = classes[classification.index(ps)]
        CLASSES[NAVIGATION_CLASS] = CLASSES.get(NAVIGATION_CLASS, 0) + 1
        for label in detections[0]["labels"]:
            pathologie = COCO_NAMES[label]
            PATHOLOGIES[pathologie] += 1
        ok, frame = video_in.read()
        if ok:
            orig_frame = frame.copy()
        if i > 200:
            break
        i += 1
    
    cv2.destroyAllWindows()
    video_in.release()
    #video_out.release()
    return CLASSES, PATHOLOGIES


def getVideos(segmentation_model, classification_model):
    s3 = boto3.resource('s3', endpoint_url='https://storage.yandexcloud.net:443')
    my_bucket = s3.Bucket('videodata')
    with open("videos_stat.txt", "w") as f:
        f.write("URL\t")
        f.write("\t".join(classes)+"\t")
        f.write("\t".join(COCO_NAMES)+ "\n")
        for obj in my_bucket.objects.all():
            print(obj.key)
            my_bucket.download_file(obj.key, "video")
            navigation, pathologies = getVideostat("video", segmentation_model, classification_model)
            f.write("{}\t".format(obj.key))
            f.write("\t".join([v for k, v in navigation.items()])+"\t")
            f.write("\t".join([v for k, v in pathologies.items()])+ "\n")
            break

now = time()
i = getVideos(segmentation_model, classification_model)
print(i / (time()-now))
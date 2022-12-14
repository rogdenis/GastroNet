import requests
import sys
import cv2
import base64
import io
import json
import torch
import random
from time import time
import numpy as np
import os.path
import urllib.parse
import torchvision.models.segmentation
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import draw_segmentation_map, filter_by_threshold, filter_nms, get_colors
from torchvision.models import resnet50

VIDEO_NAME = sys.argv[1]
TH = float(sys.argv[2])
CLASSES = ('Antrum pyloricum', 'Antrum pyloricun', 'Corpus gastricum', 'Duodenum', 'Esophagus III/III', 'Mouth', 'Oesophagus', 'Oropharynx', 'Pars cardiaca')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available

PATH = "best.pt"
if len(sys.argv) > 3: PATH = sys.argv[3]
checkpoint = torch.load(PATH)

#LOAD SEGMENTATION MODEL
PATH = "best.pt"
if len(sys.argv) > 3: PATH = sys.argv[3]
checkpoint = torch.load(PATH)
segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = segmentation_model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
segmentation_model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=6)  # replace the pre-trained head with a new one
segmentation_model.load_state_dict(checkpoint['model_state_dict'])
segmentation_model.to(device)# move model to the right devic
segmentation_model.eval()

#LOAD CLASSIFICATION MODEL
PATH = "classification.pt"
if len(sys.argv) > 4: PATH = sys.argv[4]
checkpoint = torch.load(PATH)
classification_model = resnet50(num_classes=len(CLASSES))
classification_model.load_state_dict(checkpoint['model_state_dict'])
classification_model.to(device)# move model to the right devic
classification_model.eval()

CLASSES = ('Antrum pyloricum', 'Antrum pyloricun', 'Corpus gastricum', 'Duodenum', 'Esophagus III/III', 'Mouth', 'Oesophagus', 'Oropharynx', 'Pars cardiaca')
SCORES = [0] * len(CLASSES)
A = 0.2
def drawClassification(frame, prediction):
    prediction = prediction.cpu().numpy().tolist()
    ps = max(prediction)
    pc = CLASSES[prediction.index(ps)]
    for i in range(len(SCORES)):
        SCORES[i] = A * prediction[i] + (1-A) * SCORES[i]
    s = max(SCORES)
    c = CLASSES[SCORES.index(s)]
    if c == 'Oropharynx': c = 'Oesophagus'
    elif c == 'Oesophagus': c = 'Oropharynx'
    text = "{}: {}".format(c, round(s,3))
    frame = cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    text = "{}: {}".format(pc, round(ps,3))
    pred_color = (255, 255, 255)
    if pc != c:
        pred_color = (0, 0, 255)
    frame = cv2.putText(frame, text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, pred_color, 2)
    return frame


def getFrames(segmentation_model, classification_model):
    COLORS, coco_names = get_colors()
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1)
    session.mount('http://', HTTPAdapter(max_retries=retries))
    softmax = torch.nn.Softmax(dim=0)
    video_in = cv2.VideoCapture(VIDEO_NAME)
    ok, frame = video_in.read()
    orig_frame = frame.copy()
    height,width,layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out=cv2.VideoWriter('detections_' + VIDEO_NAME, fourcc, 20.0, (2 * width,height))
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
            classification = softmax(classification_model(image)[0][:len(CLASSES)])
        predict = draw_segmentation_map(frame, detections[0], COLORS, coco_names)
        predict = drawClassification(predict, classification)
        result = np.hstack([orig_frame,predict])
        video_out.write(result)
        ok, frame = video_in.read()
        if ok:
            orig_frame = frame.copy()
        i += 1
    
    cv2.destroyAllWindows()
    video_in.release()
    video_out.release()
    return i

now = time()
i = getFrames(segmentation_model, classification_model)
print(i / (time()-now))
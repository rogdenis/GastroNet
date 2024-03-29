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
from utils import SegmentationDataset, filter_pathologies_by_research_type, filter_events_by_research_type
from torchvision.models import resnet50

DATASET = 'dataset20231029'
VIDEO_NAME = sys.argv[1]
TH = float(sys.argv[2])
VIDEO_TYPE = sys.argv[3]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available


train = SegmentationDataset(DATASET, 'annotations_coco.json')
print(train.cats)

#LOAD SEGMENTATION MODEL
PATH = "best_segmentation.pt"
if len(sys.argv) > 4: PATH = sys.argv[4]
checkpoint = torch.load(PATH)
segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = segmentation_model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
segmentation_model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=1+len(train.cats))  # replace the pre-trained head with a new one
segmentation_model.load_state_dict(checkpoint['model_state_dict'])
segmentation_model.to(device)# move model to the right devic
segmentation_model.eval()

classes = [
        "Mouth",
        "Oropharynx",
        "Esophagus",
        "Corpus gastricum",
        "Antrum pyloricum",
        "Duodenum",
        "Pathologie",
        "Anus",
        "Rectum",
        "Sigmoid colon",
        "Descending colon",
        "Left colic flexure",
        "Transverse colon",
        "Right colic flexure",
        "Ascending colon",
        "Appendix",
        "Void"]

#LOAD CLASSIFICATION MODEL
PATH = "classification.pt"
if len(sys.argv) > 5: PATH = sys.argv[5]
checkpoint = torch.load(PATH)
classification_model = resnet50(num_classes=len(classes))
classification_model.load_state_dict(checkpoint['model_state_dict'])
classification_model.to(device)# move model to the right devic
classification_model.eval()


SCORES = [0] * len(classes)
A = 0.2
def drawClassification(frame, prediction, debug=False):
    prediction = prediction.cpu().numpy().tolist()
    ps = max(prediction)
    pc = classes[prediction.index(ps)]
    for i in range(len(SCORES)):
        SCORES[i] = A * prediction[i] + (1-A) * SCORES[i]
    s = max(SCORES)
    c = classes[SCORES.index(s)]
    text = c
    if c == 'Transverse colon':
        print(prediction)
        raise
    #text = "smoothed: {}: {}".format(c, round(s,3))
    frame = cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if debug:
        text = "raw: {}: {}".format(pc, round(ps,3))
        pred_color = (255, 255, 255)
        if pc != c:
            pred_color = (0, 0, 255)
        frame = cv2.putText(frame, text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, pred_color, 2)
    return frame


def getFrames(segmentation_model, classification_model):
    COLORS, coco_names = get_colors(DATASET, 'annotations_coco.json')
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1)
    session.mount('http://', HTTPAdapter(max_retries=retries))
    softmax = torch.nn.Softmax(dim=0)
    video_in = cv2.VideoCapture(VIDEO_NAME)
    ok, frame = video_in.read()
    orig_frame = frame.copy()
    height,width,layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_file = os.path.splitext(VIDEO_NAME.split('/')[-1])[0]
    video_out=cv2.VideoWriter('detections_' + out_file + '.mp4', fourcc, 20.0, (width,height))
    print('detections_' + out_file + '.mp4')
    scores = []
    switches = 0
    pred = None
    i = 0
    while ok:
        image = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
        image = image.swapaxes(1, 3).swapaxes(2, 3).to(device)
        with torch.no_grad():
            detections = segmentation_model(image)
            detections = filter_pathologies_by_research_type(detections, coco_names, VIDEO_TYPE)
            detections = filter_nms(detections)
            detections = filter_by_threshold(detections, TH)
            #classification = classification_model(image)
            #classification = softmax(classification_model(image)[0][:len(classes)])
            classification = classification_model(image)[0][:len(classes)]
            classification = filter_events_by_research_type(classification, classes, VIDEO_TYPE)
        predict = draw_segmentation_map(frame, detections[0], COLORS, coco_names)
        predict = drawClassification(predict, classification)
        #result = np.hstack([orig_frame,predict])
        video_out.write(predict)
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
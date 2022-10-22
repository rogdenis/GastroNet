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
from utils import draw_segmentation_map, filter_by_threshold, filter_nms

VIDEO_NAME = sys.argv[1]
TH = sys.argv[2]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=6)  # replace the pre-trained head with a new one
PATH = "best.pt"
if len(sys.argv) > 3: PATH = sys.argv[3]
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint['epoch'])
model.to(device)# move model to the right devic
model.eval()

SEGMENTATION_THRESHOLD = 0.1
SEGMENTATION_PADDING = 60

CLASSIFICATION_PADDING = 30

COLORS = {
'Reflux esophagitis -La-A-': tuple(random.randint(0,255) for x in range(3)),
'Corpus gastricum': tuple(random.randint(0,255) for x in range(3)),
'Polyp -type Is-': tuple(random.randint(0,255) for x in range(3)),
'Atrophic superficial gastritis': tuple(random.randint(0,255) for x in range(3)),
'Mouth': tuple(random.randint(0,255) for x in range(3)),
'Antrum pyloricum': tuple(random.randint(0,255) for x in range(3)),
'Esophagus': tuple(random.randint(0,255) for x in range(3)),
'Duodenum': tuple(random.randint(0,255) for x in range(3))
}

TYPES = {
    "SEGMENTATION": {
        "KEY": "dKmUZU6MLwX9R504E4M1",
        "PREFIX": "https://outline.roboflow.com/instance-d8qr5/1"
    },
    "CLASSIFICATION": {
        "KEY": "YQYAOWDsYtkXsusKdvxd",
        "PREFIX": "https://classify.roboflow.com/endo-navi-gastro-v-1.0/1"
    }
}
SCORES = []

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[104, 500, 502, 503, 504] )
session.mount('https://', HTTPAdapter(max_retries=retries))

def getPrediction(fname, jpg_as_text, img_name, prediction_type):
    try:
        with open(fname) as f:
            data = json.loads(f.read())
    except:
        segmentation_url = "{}?api_key={}&name={}.jpg".format(
            TYPES[prediction_type]["PREFIX"],
            TYPES[prediction_type]["KEY"],
            img_name)
        r = session.post(segmentation_url, data=jpg_as_text, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
        data = r.json()
    with open(fname,'w') as f:
        f.write(json.dumps(data,indent=4))
    return data


def drawSegmentation(frame, jpg_as_text, count, img_name):
    fname = 'predictions/{}_prediction_segmentation4_{}.json'.format(VIDEO_NAME, count)
    data = getPrediction(fname, jpg_as_text, img_name, "SEGMENTATION")
    vertical = SEGMENTATION_PADDING
    detected = False
    for obj in data["predictions"]:
        confidence = obj['confidence']
        if confidence < SEGMENTATION_THRESHOLD:
            continue
        text = obj['class'] + ': ' + str(confidence)
        color = COLORS[obj['class']]
        center = tuple((10,vertical))
        vertical += 30
        points = []
        if len(obj["points"]) > 0:
            for point in obj["points"]:
                points.append([point["x"], point["y"]])
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            isClosed = True
            thickness = 2
            frame = cv2.polylines(frame, [pts],
                                    isClosed, color, thickness)
            frame = cv2.putText(frame, text, center,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def drawClassification(frame, jpg_as_text, count, img_name):
    fname = 'predictions/{}_prediction_classification_{}.json'.format(VIDEO_NAME, count)
    data = getPrediction(fname, jpg_as_text, img_name, "CLASSIFICATION")
    score = {k: v['confidence'] for k, v in data['predictions'].items()}
    predictions =  {k: 0 for k, v in data['predictions'].items()}
    SCORES.insert(0, score)
    if len(SCORES) > 20:
        SCORES.pop(-1)
    i = 0
    for s in SCORES:
        for prediction, v in s.items():
            predictions[prediction] += v / sum(s.values())
        i += 1
    prediction_score = {k: v for k, v in sorted(predictions.items(), key=lambda item: -item[1])}
    pred_i = 0
    for k, v in prediction_score.items():
        if k == 'Oropharynx': k = 'Oesophagus'
        elif k == 'Oesophagus': k = 'Oropharynx'
        break
    text = "{}: {}".format(k, round(v,3))
    frame = cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def getFrames(model):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1)
    session.mount('http://', HTTPAdapter(max_retries=retries))

    video_in = cv2.VideoCapture(VIDEO_NAME)
    ok, frame = video_in.read()
    orig_frame = frame.copy()
    height,width,layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out=cv2.VideoWriter('detections_' + VIDEO_NAME, fourcc, 20.0, (2 * width,height))
    count = 0
    scores = []
    switches = 0
    pred = None
    i = 0
    while ok and count <= 400:
        image = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0)
        image = image.swapaxes(1, 3).swapaxes(2, 3)
        with torch.no_grad():
            outputs = model(image.to(device))
        outputs = filter_nms(outputs)
        filtered_outs = filter_by_threshold(outputs, TH)
        if len(filtered_outs[0]['labels']) > 0:
            print(i)
        predict = draw_segmentation_map(frame, filtered_outs[0])
        result = np.hstack([orig_frame,predict])
        video_out.write(result)
        ok, frame = video_in.read()
        if ok:
            orig_frame = frame.copy()
        i += 1
    
    cv2.destroyAllWindows()
    video_in.release()
    video_out.release()

now = time()
getFrames(model)
print(time()-now, (time()-now) / 400)
import json
import requests
import shutil
import sys
import cv2
import os
import numpy as np
from PIL import Image
from collections import Counter
from imagehash import colorhash
#from random import random
from requests.auth import HTTPBasicAuth

DATASETDIR = "dataset20231029"
if not os.path.isdir(DATASETDIR):
    os.makedirs(DATASETDIR)

URLS = ['https://storage.yandexcloud.net/cvproject/labeled_data/next_ten_x8mio8__.json',
        'https://storage.yandexcloud.net/cvproject/labeled_data/first10_35gvxn__.json',
        'https://storage.yandexcloud.net/cvproject/labeled_data/colono2__.json',
        'https://storage.yandexcloud.net/cvproject/labeled_data/batch_bv21ig_dain080888.json']

TASKS = ["639dd94668ba2c0058cfb79c",
"639dd94668ba2c0058cfb7a3",
"639dd94668ba2c0058cfb7a2",
"639dd94668ba2c0058cfb79e",
"639dd94668ba2c0058cfb7a0",
"639dd94668ba2c0058cfb79d",
"639dd94668ba2c0058cfb7a4",
"639dd94668ba2c0058cfb79f",
"63d7c9103dd0305f59c1d966",
"63d7c9103dd0305f59c1d970",
"63d7c9103dd0305f59c1d96e",
"63d7c9103dd0305f59c1d96c",
"63d7c9103dd0305f59c1d96d",
"63d7c9103dd0305f59c1d96b",
"63d7c9103dd0305f59c1d96f",
"63d7c9103dd0305f59c1d967",
"63d7c9103dd0305f59c1d968",
"64b83136f25ef0055d99662d",
"65103452c07bcdca6ed02796",
"65103452c07bcdca6ed0278e",
"65103452c07bcdca6ed0279d",
"65103452c07bcdca6ed027a5",
"65103452c07bcdca6ed02791",
"65103452c07bcdca6ed0279f",
"65103452c07bcdca6ed02794",
"65103452c07bcdca6ed0279b"
]

PATHOLOGIES = [
    "Kartsenoma cords",
    "Reflux esophagitis",
    "Esophagus Barretta",
    "Eosinophilic esophagitis",
    "Varicose veins",
    "Achalasia",
    "Hiatal hernia",
    "Kartsenoma",
    "Non-atrophic superficial gastritis",
    "Atrophic superficial gastritis",
    "Hp. pylori associated gastritis",
    "Erosive gastritis",
    "Duodenogastric reflux",
    "Xanthoma",
    "Polyp",
    "Uncer",
    "Diverticulum",
    "Hemorrhoids"
]

PARTS = {
    "Mouth": 0,
    "Oropharynx": 0,
    "Esophagus": 0.5,
    "Corpus gastricum": 1,
    "Antrum pyloricum": 0,
    "Duodenum": 0,
    "Pathologie": 0,
    "Anus": 0,
    "Rectum": 0,
    "Sigmoid colon": 0,
    "Descending colon": 0,
    "Left colic flexure": 0,
    "Transverse colon": 0,
    "Right colic flexure": 0,
    "Ascending colon": 0,
    "Appendix": 0,
    "Void": 0
}

STAT = {
    "Mouth": 0,
    "Oropharynx": 0,
    "Esophagus": 0,
    "Corpus gastricum": 0,
    "Antrum pyloricum": 0,
    "Duodenum": 0,
    "Pathologie": 0,
    "Anus": 0,
    "Rectum": 0,
    "Sigmoid colon": 0,
    "Descending colon": 0,
    "Left colic flexure": 0,
    "Transverse colon": 0,
    "Right colic flexure": 0,
    "Ascending colon": 0,
    "Appendix": 0,
    "Void": 0
}

hash_stat = set()

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename


def extract_frame(vidcap, start, end, label, task_id, freq = 10):
    STAT[label] += end - start
    last_hash = None
    success, image = vidcap.read()
    img_hash = colorhash(Image.fromarray(image), binbits=7)
    count = start
    images_to_labels = []
    #print(label, prob)
    while success and count < end:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', gray, encode_param)
        compression_rate = sys.getsizeof(encimg) / sys.getsizeof(gray)
        if (last_hash is None or (img_hash - last_hash > freq * PARTS[label] and img_hash not in hash_stat))\
            and compression_rate > 0.05:
            filename = "{}_{}.png".format(task_id, count)
            path = os.path.join(DATASETDIR, filename)
            cv2.imwrite(path, image)
            images_to_labels.append(','.join((filename, label)))
            last_hash = img_hash
        hash_stat.add(str(img_hash))
        success, image = vidcap.read()
        img_hash = colorhash(Image.fromarray(image), binbits=7)
        count += 1
    return count, images_to_labels

def extract_frames(vidcap, frames, destination, width, height):
    success, image = vidcap.read()
    images = []
    frame_id = 0
    for filename, props in frames.items():
        frame = props["frame"]
        image_id = props["image_id"]
        while frame_id < frame:
            success, image = vidcap.read()
            frame_id += 1
        path = os.path.join(destination, filename)
        cv2.imwrite(path, image)
        images.append({"id": image_id, "file_name": filename, "width": width, "height": height})
    return images

def create_coco(pathologies):
    COCO = {"images":[],"annotations":[],"categories": []}
    COCO["categories"].append({"id": 0, "name": "Pathology", "supercategory": "none"})
    i = 1
    for pathologie in pathologies:
        COCO["categories"].append({"id": i, "name": pathologie, "supercategory": "Pathology"})
        i += 1
    return COCO

# url = "https://api.scale.com/v1/tasks"
# headers = {"Accept": "application/json"}
# auth = HTTPBasicAuth('test_a02653c3326a4da9bfabb3fadd61873b', '') # No password
# response = requests.request("GET", url, headers=headers, auth=auth)
#download batch_file

data = []
tasks = set()
if os.path.isfile(os.path.join(DATASETDIR,'data.json')):
    with open(os.path.join(DATASETDIR,'data.json')) as f:
        data = json.load(f)
    for task in data:
        tasks.add(task["task_id"])
for url in URLS:
    r = requests.get(url)
    for task in r.json():
        if task["task_id"] not in tasks:
            data.append(task)
            print("add {} from {}".format(task["task_id"], url))

COCO = create_coco(PATHOLOGIES)
NAVIGATION = []
annotation_id = 0
images_with_annotations = 0
for video in data:
    task_id = video["task_id"]
    if task_id not in TASKS:
        print("skip {}".format(task_id))
        continue
    if video.get('completed_at') is None:
        continue
    #download video
    if 'originalUrl' in video['metadata']:
        video_url = video['metadata']['originalUrl']
        videofilename = video_url.split('/')[-1]
    else:
        video_url = video['attachmentS3Downloads'][-1]['s3URL']
        videofilename = video['metadata']['filename']
    meta = video["metadata"]["video"]
    width = video["metadata"]["video"]["resolution"]["w"]
    height = video["metadata"]["video"]["resolution"]["h"]
    #r = requests.get(video_url)
    print(videofilename)
    if not os.path.isfile(videofilename):
        print("download video")
        download_file(video_url, videofilename)
    else:
        print("video downloaded, skip")
    response = video["response"]
    #navigation
    if 'data' not in response['events']:
        r = requests.get(response['events']['url'])
        events = r.json()
        response['events']['data'] = events
    else:
        events = response['events']['data']
    start = 0
    label = "Void"
    label_ranged = False
    vidcap = cv2.VideoCapture(videofilename)
    for event in events:
        if event["label"] == "Pathologie":
            continue
        #print(event)
        if not label_ranged:
            #print("{} from {} until {}".format(label, start, event["start"]))
            start, end = start, event["start"]
            start, images_to_labels = extract_frame(vidcap, start, end, label, task_id)
            NAVIGATION += images_to_labels
        label = event["label"]
        if event["type"] == "range":
            #print("{} from {} until {}".format(label, event["start"], event["end"]))
            start, end = event["start"], event["end"]
            start, images_to_labels = extract_frame(vidcap, start, end, label, task_id)
            label_ranged = True
        else:
            label_ranged = False
        NAVIGATION += images_to_labels
    
    #segmentation
    if 'data' not in response['annotations']:
        print("download data for {}".format(task_id))
        r = requests.get(response['annotations']['url'])
        annotations = r.json()
        response['annotations']['data'] = annotations
    else:
        annotations = response['annotations']['data']
    frames = {}
    vidcap = cv2.VideoCapture(videofilename)
    for key, annotation in annotations.items():
        imagefilename = "{}_{}.png".format(task_id, annotation["frames"][0]["key"])
        if imagefilename not in frames:
            frames[imagefilename] = {
                "image_id":  images_with_annotations,
                "frame": annotation["frames"][0]["key"]
            }
            images_with_annotations += 1
        obj = {}
        obj["id"] = annotation_id
        obj["image_id"] = frames[imagefilename]["image_id"]
        obj["category_id"] = PATHOLOGIES.index(annotation["label"]) + 1
        obj["segmentation"] = [[]]
        for vertice in annotation["frames"][0]["vertices"]:
            obj["segmentation"][0] += [vertice["x"],vertice["y"]]
        COCO["annotations"].append(obj)
        annotation_id += 1
    COCO["images"] += extract_frames(vidcap, frames, DATASETDIR, width, height)
print(json.dumps(STAT,indent = 4))


with open(os.path.join(DATASETDIR,"annotations_coco.json"),"w") as f:
    f.write(json.dumps(COCO, indent = 4))
with open(os.path.join(DATASETDIR,"navigation.csv"),"w") as f:
    f.write('\n'.join(NAVIGATION))
with open(os.path.join(DATASETDIR,"data.json"), "w") as f:
    f.write(json.dumps(data,indent=4))
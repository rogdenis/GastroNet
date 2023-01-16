import json
import requests
import shutil
import sys
import cv2
import os
from random import random
from requests.auth import HTTPBasicAuth

DATASETDIR = "dataset20230116"

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
    "Uncer"
]

PARTS = {
    "Antrum pyloricum": 122/329,
    "Corpus gastricum": 1,
    "Duodenum": 30/329,
    "Esophagus": 106/329,
    "Mouth": 12/329,
    "Oropharynx": 32/329,
    "Void": 49/329
}

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename


def extract_frame(vidcap, start, end, label, task_id, freq = 0.05):
    success, image = vidcap.read()
    count = start
    images_to_labels = []
    prob = min(1, freq * 1.0 / PARTS[label])
    #print(label, prob)
    while success and count < end:
        if random() < prob:
            filename = "{}_{}.png".format(task_id, count)
            path = os.path.join(DATASETDIR, filename)
            cv2.imwrite(path, image)
            images_to_labels.append(','.join((filename, label)))
        success, image = vidcap.read()
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

URL = 'https://storage.yandexcloud.net/cvproject/tasks_fixed.json'
r = requests.get(URL)
data = r.json()

COCO = create_coco(PATHOLOGIES)
NAVIGATION = []
annotation_id = 0
images_with_annotations = 0
for video in data:
    task_id = video["task_id"]
    if video.get('completed_at') is None:
        continue
    #download video
    video_url = video.get('attachmentS3Downloads')[1]['s3URL']
    videofilename = video['metadata']['filename']
    meta = video["metadata"]["video"]
    width = video["metadata"]["video"]["resolution"]["w"]
    height = video["metadata"]["video"]["resolution"]["h"]
    #r = requests.get(video_url)
    print(videofilename)
    if not os.path.isfile(videofilename):
        download_file(video_url, videofilename)
    
    response = video["response"]
    #navigation
    r = requests.get(response['events']['url'])
    events = r.json()
    start = 0
    label = "Void"
    label_ranged = False
    vidcap = cv2.VideoCapture(videofilename)
    for event in events:
        if event["label"] == "Pathologie":
            continue
        #print(event)
        if not label_ranged:
            print("{} from {} until {}".format(label, start, event["start"]))
            start, end = start, event["start"]
            start, images_to_labels = extract_frame(vidcap, start, end, label, task_id)
            NAVIGATION += images_to_labels
        label = event["label"]
        if event["type"] == "range":
            print("{} from {} until {}".format(label, event["start"], event["end"]))
            start, end = event["start"], event["end"]
            start, images_to_labels = extract_frame(vidcap, start, end, label, task_id)
            label_ranged = True
        else:
            label_ranged = False
        NAVIGATION += images_to_labels
    
    #segmentation
    r = requests.get(response['annotations']['url'])
    annotations = r.json()
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

with open(os.path.join(DATASETDIR,"annotations_coco.json"),"w") as f:
    f.write(json.dumps(COCO, indent = 4))
with open(os.path.join(DATASETDIR,"navigation.csv"),"w") as f:
    f.write('\n'.join(NAVIGATION))
# print("navigation classes:")
# print(sorted(list(classes)))
import json
import requests
import shutil
import sys
import cv2
import os
from requests.auth import HTTPBasicAuth

CLASSIFICATION_DIR = "classification20230104"

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename


def extract_frame(vidcap, start, end, label, task_id, tsv, freq = 10):
    success, image = vidcap.read()
    count = start
    while success and count < end:
        if count % freq == 0:
            filename = "{}_{}.png".format(task_id, count)
            path = os.path.join(CLASSIFICATION_DIR, filename)
            cv2.imwrite(path, image)
            tsv.write("{},{}\n".format(filename, label)) 
        success, image = vidcap.read()
        count += 1
    return count

# url = "https://api.scale.com/v1/tasks"
# headers = {"Accept": "application/json"}
# auth = HTTPBasicAuth('test_a02653c3326a4da9bfabb3fadd61873b', '') # No password
# response = requests.request("GET", url, headers=headers, auth=auth)
#download batch_file

URL = 'https://storage.yandexcloud.net/cvproject/labeled_scale_1.json'
r = requests.get(URL)
data = r.json()

classes = set()
for video in data:
    task_id = video["task_id"]
    if video.get('completed_at') is None:
        continue
    #download video
    video_url = video.get('attachmentS3Downloads')[1]['s3URL']
    
    filename = video['metadata']['filename']
    meta = video["metadata"]["video"]
    print(filename)
    download_file(video_url, filename)
    vidcap = cv2.VideoCapture(filename)
    response = video["response"]
    #download events
    r = requests.get(response['events']['url'])
    events = r.json()
    start = 0
    label = "Void"
    with open(os.path.join(CLASSIFICATION_DIR,'_classes.csv'),'a') as tsv:
        for event in events:
            if event["label"] == "Pathologie":
                continue
            classes.add(event["label"])
            # print("{} from {} until {}".format(label, start, event["start"]))
            start = extract_frame(vidcap, start, event["start"], label, task_id, tsv)
            label = event["label"]
            if event["type"] == "range":
                # print("{} from {} until {}".format(label, start, event["end"]))
                start = extract_frame(vidcap, start, event["end"], label, task_id, tsv)

print("navigation classes:")
print(sorted(list(classes)))
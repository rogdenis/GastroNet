from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from random import shuffle

coco = COCO('Gastro.v1i.coco-segmentation/train/_annotations.coco.json')
imgIds = coco.getImgIds()
shuffle(imgIds)
print(imgIds)
names = {}
for cat in coco.loadCats(coco.getCatIds()):
    names[cat['id']] = cat["name"]
print(names)
for image_id in imgIds:
    file_name = coco.imgs[image_id]['file_name']
    ann_ids = coco.getAnnIds(imgIds=image_id)
    for ann_id in ann_ids:
        ann = coco.loadAnns(ann_id)
        mask = coco.annToMask(ann[0])
        # print(image_id, ann[0]['category_id'])
        # im = Image.fromarray(mask, '1')
        # im.save("mask.png")
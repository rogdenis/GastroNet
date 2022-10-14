import cv2
import numpy as np
import random
import torch
import os
from pycocotools.coco import COCO
import torchvision.transforms as T

imageSize = [800,600]
coco = COCO(os.path.join('/home/rogdenis/segmentation/Gastro.v1i.coco-segmentation/train','_annotations.coco.json'))
coco_names = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def loadData(trainDir, onlycats=False, cats=None, augmentation=False):
    coco = COCO(os.path.join(trainDir,'_annotations.coco.json'))
    imgIds = coco.getImgIds()
    random.shuffle(imgIds)
    if cats is None:
        cats = {v: k for k, v in dict(enumerate(coco.getCatIds())).items()}
    dataset = []
    while onlycats == False and len(imgIds) > 0:
        idx=imgIds[0]
        file_name = coco.imgs[idx]['file_name']
        img = cv2.imread(os.path.join(trainDir, file_name))
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        ann_ids = coco.getAnnIds(imgIds=idx)
        masks=[]
        labels=[]
        for ann_id in ann_ids:
            ann = coco.loadAnns(ann_id)
            mask = coco.annToMask(ann[0])
            #mask=cv2.resize(mask,imageSize,cv2.INTER_NEAREST)
            masks.append(mask)
            labels.append(cats[ann[0]['category_id']])
        num_objs = len(masks)
        imgIds.pop(0)
        if num_objs == 0: 
            continue
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        imgs=[img]
        if augmentation:
            #imgs += [T.GaussianBlur(kernel_size=(51, 91), sigma=sigma)(img) for sigma in range(2,10)]
            imgs += [T.RandomRotation(degrees=d)(img) for d in range(0,5)]
        for img in imgs:
            #print(img.shape)
            data = {}
            data["boxes"] =  boxes
            data["labels"] = torch.tensor(labels, dtype=torch.int64)#torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
            data["masks"] = masks
            dataset.append((img, data))
    return dataset, cats

def make_batch(dataset, batchSize, start):
    #print(dataset[0][1])
    batch_Imgs = []
    batch_Data = []
    while len(batch_Imgs) < batchSize:
        batch_Imgs.append(dataset[start][0])
        batch_Data.append(dataset[start][1])
        start += 1
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data, start


def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]

    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.3 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        image = cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        image = cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image
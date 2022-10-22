import cv2
import numpy as np
import random
import torch
import os
from time import time
from pycocotools.coco import COCO
import torchvision.ops
import torchvision.transforms as T
from mean_average_precision import MetricBuilder
from collections import Counter

coco = COCO(os.path.join('/home/rogdenis/GastroNet/Gastro.v1i.coco-segmentation/train','_annotations.coco.json'))
coco_names = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
coco_names.insert(0,"bg")

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def make_coco(trainDir):
    coco = COCO(os.path.join(trainDir,'_annotations.coco.json'))
    imgIds = coco.getImgIds()
    random.shuffle(imgIds)
    cats = {v: k for k, v in dict(enumerate(coco.getCatIds())).items()}
    return(coco, imgIds, cats)


def filter_nms(outputs):
    outs = []
    for output in outputs:
        out = {}
        indexes = torchvision.ops.nms(output['boxes'], output['scores'], iou_threshold=0.1)
        out['scores'] = output['scores'][indexes]
        out['masks'] = output['masks'][indexes]
        out['boxes'] = output['boxes'][indexes]
        out['labels'] = output['labels'][indexes]
        outs.append(out)
    return outs
    

def filter_by_threshold(outputs, threshold):
    outs = []
    for output in outputs:
        out = {}
        scores = list(output['scores'].detach().cpu().numpy())
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        out['scores'] = output['scores'][:thresholded_preds_count]
        masks = (output['masks']>0.6).squeeze().detach().cpu().numpy()
        out['masks'] = masks[:thresholded_preds_count]
        out['boxes'] = output['boxes'][:thresholded_preds_count]
        out['labels'] = output['labels'][:thresholded_preds_count]
        outs.append(out)
    return outs

def convert_to_array(detections, targets):
    preds_boxes = detections['boxes'].cpu().numpy()
    preds_scores = detections['scores'].cpu().numpy()
    preds_labels = detections['labels'].cpu().numpy()
    preds = np.column_stack((preds_boxes, preds_labels, preds_scores))

    gt_boxes = targets['boxes'].cpu().numpy()
    gt_labels = targets['labels'].cpu().numpy()
    zeros = np.zeros(len(gt_boxes))
    gt = np.column_stack((gt_boxes,gt_labels, zeros, zeros))
    return preds, gt

def calculate_metrics(detections, targets, classes, iou):
    #print(detections)
    labels_gt = Counter(targets[:,4])
    labels_preds = Counter(detections[:,4])
    gt_num = len(targets[:,4])
    preds_num = len(detections[:,4])
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=classes)
    metric_fn.add(detections, targets)
    map_stat = metric_fn.value(iou_thresholds=iou)[iou]
    TP = 0
    FP = 0
    #print(metric_fn.value(iou_thresholds=0.5))
    for label, stat in map_stat.items():
        #print(label, stat)
        cl_tp = labels_gt[label] * stat['recall'][-1] if len(stat['recall']) > 0 else 0
        cl_fp = labels_preds[label] - cl_tp
        #print(cl_tp)
        TP += cl_tp
        FP += cl_fp
    recall = TP * 1.0 / gt_num if gt_num > 0 else 0
    precision = TP * 1.0 / preds_num if preds_num > 0 else 0
    FP = preds_num - TP
    FN = gt_num - TP
    return TP, FP, FN


def draw_segmentation_map(image, output):
    alpha = 1 
    beta = 0.3 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    # print(output['masks'])
    # print(output['labels'])
    # print(output['boxes'])
    masks, boxes, labels, scores = output['masks'], output['boxes'], output['labels'], output.get('scores', None)
    for i in range(len(output['masks'])):
        try:
            #print(boxes[i])
            #convert the original PIL image into NumPy format
            image = np.array(image)
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # apply a randon color mask to each object
            color = COLORS[random.randrange(0, len(COLORS))]
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            # convert from RGN to OpenCV BGR format
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in output['boxes'].detach().cpu()]
            image = cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                                thickness=2)
            # get the classes labels
            if scores is not None:
                label = "{:.2f}: {}".format(scores[i], coco_names[output['labels'][i]])
            else:
                label = coco_names[output['labels'][i]]
            # # put the label text above the objects
            image = cv2.putText(image , label, (boxes[i][0][0], boxes[i][0][1]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                        thickness=2, lineType=cv2.LINE_AA)
        except:
            pass
    return image
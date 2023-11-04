import cv2
import numpy as np
import random
import torch
import os
import random
import json
import torchvision.ops
from time import time
from torchvision import transforms
from pycocotools.coco import COCO
from torchvision.io import read_image
from torch.utils.data import Dataset
from collections import Counter
from mean_average_precision import MetricBuilder


def get_colors(coco_dir, annotation_file):
    coco = COCO(os.path.join(coco_dir, annotation_file))
    coco_names = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
    coco_names.insert(0,"bg")
    return np.random.uniform(0, 255, size=(len(coco_names), 3)), coco_names


class ClassificationDataset(Dataset):
    def __init__(self, img_dir, annotations_file, classes,
                seq, dataset_type,
                image_transform=None, coords_transform=None, SEED = "0"):
        self.NAVIGATION_CLASSES = classes
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.coords_transform = coords_transform
        self.indx = []
        with open(os.path.join(img_dir, annotations_file)) as f:
            for line in f:
                if next(seq) != dataset_type:
                    continue
                tabs = line.strip().split(',')
                fname = tabs[0]
                cl = self.NAVIGATION_CLASSES.index(tabs[1])
                # blur = cv2.Laplacian(cv2.imread(os.path.join(img_dir, fname)), cv2.CV_64F).var()
                # variance = np.var(cv2.imread(os.path.join(img_dir, fname)))
                # if blur > 100 or variance > 2500:
                self.indx.append((fname, cl))

    def __len__(self):
        return len(self.indx)

    def __getitem__(self, idx):
        obj = self.indx[idx]
        img_path = os.path.join(self.img_dir, obj[0])
        image = cv2.imread(img_path)#read_image(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_transform:
            transformed = self.image_transform(image=image)
            image = transformed['image']
        if self.coords_transform:
            transformed = self.coords_transform(image=image)
            image = transformed['image']
            image = torch.as_tensor(image, dtype=torch.float32)
        data = obj[1]
        image = torch.as_tensor(image, dtype=torch.float32).swapaxes(0, 2).swapaxes(1, 2)
        return image, data

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, annotations_file, cats=None,
                 image_transform=None, coords_transform=None,
                 bg=True, seq=None, dtype="train", empty_rate=100):
        self.img_dir = img_dir
        self.coco = COCO(os.path.join(img_dir, annotations_file))
        if cats is None:
            self.cats = {v: k+1 for k, v in dict(enumerate(self.coco.getCatIds())).items()}
        else:
            self.cats = cats
        self.image_transform = image_transform
        self.coords_transform = coords_transform
        self.bg = bg
        self.indx = []
        self.dtype = dtype
        if seq is not None:
            for idx in range(len(self.coco.getImgIds())):
                if next(seq) != dtype:
                        continue
                self.indx.append(idx)
        else:
            self.indx = list(range(len(self.coco.getImgIds())))
        # self.balanced_index = []
        # empty = []
        # for idx in range(len(self.coco.getImgIds())):
        #     print(idx)
        #     if len(self.coco.getAnnIds(imgIds=idx)) > 0:
        #         self.balanced_index.append(idx)
        #     else:
        #         empty.append(idx)
        # K = int(min(len(empty), len(self.balanced_index) * empty_rate))
        # indices = random.sample(range(len(empty)), K)
        # self.balanced_index += [empty[i] for i in sorted(indices)]
        # random.shuffle(self.balanced_index)
        # print("empty", len(indices), "all", len(self.balanced_index))

    def __len__(self):
        return len(self.indx)
        #return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        idx = self.indx[idx]
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        num_objs = len(ann_ids)
        img_path = os.path.join(self.img_dir, self.coco.imgs[idx]['file_name'])
        image = cv2.imread(img_path)#read_image(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_transform:
            transformed = self.image_transform(image=image)
            image = transformed['image']
        masks = []
        labels = []
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        if self.bg:
            num_objs += 1
            boxes = torch.zeros([num_objs,4], dtype=torch.float32)
            bg = np.zeros(image.shape[:2],dtype=np.uint8)
            masks = [bg]
            labels = [0]
            boxes[0] = torch.tensor([0, 0, image.shape[1], image.shape[0]])
        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)
            mask = self.coco.annToMask(ann[0])
            masks.append(mask)
            labels.append(self.cats[ann[0]['category_id']])
        if self.coords_transform:
            transformed = self.coords_transform(image=image, masks=masks)
            image = transformed['image']
            masks = transformed['masks']
            image = torch.as_tensor(image, dtype=torch.float32)
        for i in range(int(self.bg), num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.float32)
        image = torch.as_tensor(image, dtype=torch.float32).swapaxes(0, 2).swapaxes(1, 2)
        data = {}
        data["boxes"] =  boxes
        data["labels"] = torch.tensor(labels, dtype=torch.int64)#torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        return image, data


def collate_fn(batch):
    return tuple(zip(*batch))


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
        thresholded_preds_indices = [scores.index(i) for i in scores if i > threshold]
        out['scores'] = output['scores'][thresholded_preds_indices]
        masks = output['masks']>0.6
        out['masks'] = masks[thresholded_preds_indices]
        out['boxes'] = output['boxes'][thresholded_preds_indices]
        out['labels'] = output['labels'][thresholded_preds_indices]
        outs.append(out)
    return outs


def filter_events_by_research_type(outputs, classes, videotype):
    cl = outputs.detach().cpu().numpy()
    with open('restrictions.json') as f:
        restrictions = json.load(f)
        impossible_events_index = [classes.index(c) for c in classes if c not in restrictions[videotype]['events']]
    outputs[impossible_events_index] = -1000.0
    return outputs


def filter_pathologies_by_research_type(outputs, coco_names, videotype):
    outs = []
    with open('restrictions.json') as f:
        restrictions = json.load(f)
        possible_detections = restrictions[videotype]['detections']
    for output in outputs:
        out = {}
        labels = list(output['labels'].detach().cpu().numpy())
        selected_detections = [labels.index(l) for l in labels if coco_names[l] in possible_detections]
        out['scores'] = output['scores'][selected_detections]
        masks = output['masks']>0.6
        out['masks'] = masks[selected_detections]
        out['boxes'] = output['boxes'][selected_detections]
        out['labels'] = output['labels'][selected_detections]
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


def draw_segmentation_map(image, output, COLORS, coco_names, draw_poly=True):
    alpha = 1 
    beta = 0.4 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    # print(output['masks'])
    # print(output['labels'])
    # print(output['boxes'])
    masks, boxes, labels, scores = output['masks'], output['boxes'], output['labels'], output.get('scores', None)
    for i in range(len(output['masks'])):
        #print(boxes[i])
        #convert the original PIL image into NumPy format
        image = np.array(image)
        mask = masks[i].squeeze().detach().cpu().numpy()
        red_map = np.zeros_like(mask).astype(np.uint8)
        green_map = np.zeros_like(mask).astype(np.uint8)
        blue_map = np.zeros_like(mask).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[mask == 1], green_map[mask == 1], blue_map[mask == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # convert from RGN to OpenCV BGR format
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        if draw_poly:
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
    return image
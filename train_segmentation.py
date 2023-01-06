#DONE #https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
#DONE #https://zenodo.org/record/4736111/files/LabPicsChemistry.zip?download=1
#DONE #https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code
#DONE валидация
#DONE метрики
#DONE https://towardsdatascience.com/a-comprehensive-guide-to-image-augmentation-using-pytorch-fb162f2444be - аугментация
#DONE GPU
#DONE тензорборда
#DONE https://pytorch.org/tutorials/beginner/saving_loading_models.html - загрузка моделей
#DONE https://stackoverflow.com/questions/67295494/correct-validation-loss-in-pytorch criterion
#[I 2022-10-30 06:40:29,961] Trial 14 finished with value: 0.06895320117473602 and parameters: {'batch': 2, 'lr': 8.16519294186843e-05, 'WD': 6.745022008823777e-05}. Best is trial 13 with value: 0.08766011148691177.
#BEST [I 2022-10-30 05:58:52,136] Trial 13 finished with value: 0.08766011148691177 and parameters: {'batch': 1, 'lr': 7.996305978961509e-05, 'WD': 5.727217598883216e-05}. Best is trial 13 with value: 0.08766011148691177.

import random
import json
import numpy as np
import cv2
import os
import torch
import sys
import datetime
import optuna
from optuna.trial import TrialState
import torchvision.models.segmentation
import albumentations as A
from torchvision import transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from utils import filter_by_threshold, filter_nms, calculate_metrics, convert_to_array, SegmentationDataset, collate_fn
from mean_average_precision import MetricBuilder

DATE = datetime.datetime.now().strftime("%Y%m%d%H%M")
print(DATE)
BEST = -1

image_transform = A.Compose([
    A.MotionBlur(p=0.5),
    A.Defocus(p=0.5)
])

coords_transform = A.Compose([
    A.Affine(shear=(-5, 5), p=0.5),
    A.Flip(p=0.5)
])
train = SegmentationDataset('dataset', 'annotations_coco.json',
    image_transform=image_transform,
    coords_transform=coords_transform,
    empty_rate=100,
    bg=True)
valid = SegmentationDataset('dataset', 'annotations_coco.json', cats=train.cats, bg=True)
print("train len", train.__len__())
print("train cats", train.cats)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def objective(trial):
    global BEST
    val_batch = 1
    #tb4_vb1_lr0.00014_wd0.00055_er100
    train_batch = trial.suggest_int("batch", 4, 4, log=False)
    LR = trial.suggest_float("lr", 0.00014, 0.00014, log=True)#0.0001
    WD = trial.suggest_float("WD", 0.00055, 0.00055, log=False)#0.001
    ER = 100

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # load an instance segmentation model pre-trained pre-trained on COCO
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=1+len(train.cats))  # replace the pre-trained head with a new one
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay = WD)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=20,
                                                    gamma=0.1)
    train_dataloader = DataLoader(train, batch_size=train_batch,
    shuffle=True,
    collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid, batch_size=val_batch, shuffle=False, collate_fn=collate_fn) 

    pth = 'runs/{}/tb{}_vb{}_lr{:.5f}_wd{:.5f}_er{}'.format(
        DATE,
        train_batch,
        val_batch,
        LR,
        WD,
        ER
        )
    print(pth)
    writer = SummaryWriter(pth)

    start_epoch = 0
    if len(sys.argv) > 1 and sys.argv[1] == 'continue':
        if len(sys.argv) > 2:
            PATH = sys.argv[2]
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        BEST = checkpoint['metric']

    for epoch in range(start_epoch, 100):
        model.train()
        i = 0
        for images, targets in train_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            #writer.add_scalars('training-loss_parts', loss_dict, epoch * len(train_dataloader) + i)
            writer.add_scalar('training-loss', losses * 1.0 / train_batch, epoch * len(train_dataloader) + i)
            losses.backward()
            optimizer.step()
            i += 1
            if i % 2000 == 0:
                print(i,'loss:', losses.item())

        #VALIDATE
        #model.eval()
        main_metric = 0
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1+len(train.cats))
        losses = 0
        with torch.no_grad():
            for images, targets in valid_dataloader:
                model.train()
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses += sum(loss for loss in loss_dict.values())
                model.eval()
                outputs = model(images)
                outputs = filter_nms(outputs)
                filtered_outs = filter_by_threshold(outputs, 0)
                for out, target in zip(filtered_outs, targets):
                    preds, gt = convert_to_array(out, target)
                    metric_fn.add(preds, gt)

        mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        total_loss = losses / len(valid_dataloader)
        trial.report(mAP, epoch)
        #print('epoch {} map: {} loss {}'.format(epoch, mAP, total_loss))
        
        writer.add_scalar('validate-map', mAP, epoch)
        writer.add_scalar('validate-loss', total_loss, epoch)
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,
                'metric': mAP,
                'params': pth
                }
        #torch.save(checkpoint, "last_{}.pt".format(pth).replace("/",""))

        if mAP > BEST:
            BEST = mAP
            torch.save(checkpoint, "best_segmentation.pt".format(pth.replace("/","")))

        lr_scheduler.step()

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mAP

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
        pruner=optuna.pruners.ThresholdPruner(
            lower=1e-3, n_warmup_steps=5, interval_steps=5
        ))
    study.optimize(objective, n_trials=1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    with open('opimization.txt',"w") as f:
        f.write("Study statistics:\n")
        f.write("  Number of finished trials:{}".format(len(study.trials)))
        f.write("  Number of pruned trials:{}".format(len(pruned_trials)))
        f.write("  Number of complete trials:{}".format(len(complete_trials)))

        f.write("Best trial:\n")
        trial = study.best_trial

        f.write("  Value:{}".format(trial.value))

        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write("    {}: {}".format(key, value))
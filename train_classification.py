#https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
import torch
import torchvision
import optuna
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A
from random import choices
from random import seed
from optuna.trial import TrialState
from utils import ClassificationDataset, collate_fn
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter

population = ["train", "valid"]
weights = [0.8, 0.2]
seed(0)
seq = iter(choices(population, weights, k=10 ** 5))

classes = [
    'Antrum pyloricum',
    'Corpus gastricum',
    'Duodenum',
    'Esophagus',
    'Mouth',
    'Oropharynx',
    'Void']

image_transform = A.Compose([
    A.MotionBlur(p=0.5),
    A.Defocus(p=0.5)
])

coords_transform = A.Compose([
    A.Affine(p=0.5),
    A.Flip(p=0.5)
])

trainset = ClassificationDataset('classification20230104', '_classes.csv', classes,
    seq, "train",
    image_transform=image_transform,
    coords_transform=coords_transform,
)

valid = ClassificationDataset('classification20230104', '_classes.csv', classes,
    seq, "valid",
    image_transform=None,
    coords_transform=None,
)

print(len(trainset))
print(len(valid))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BEST = -1

def objective(trial):
    global BEST
#     {'batch': 16, 'lr': 3.41968662651
# 88484e-05, 'WD': 6.374732293412834e-05}
    train_batch = trial.suggest_int("batch", 16, 16, log=False)
    LR = trial.suggest_float("lr", 3.4e-05, 3.4e-05, log=True)#0.0001
    WD = trial.suggest_float("WD", 6.37e-05, 6.37e-05, log=False)#0.001
    pth = 'classification_log_2/{}_{}_{}'.format(train_batch, LR, WD)
    print(pth)
    writer = SummaryWriter(pth)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid, batch_size=train_batch, shuffle=False)
    model = resnet50(num_classes=len(classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay = WD)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(40):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('training-loss', loss, epoch * len(trainloader) + i)

        # correct_pred = {classname: 0 for classname in classes}
        # total_pred = {classname: 0 for classname in classes}
        # total_label = {classname: 0 for classname in classes}
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in validloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if prediction == label:
                        # correct_pred[classes[label]] += 1
                        correct += 1
                    total += 1
                    # total_pred[classes[prediction]] += 1
                    # total_label[classes[label]] += 1

        lr_scheduler.step()

        accuracy = 100 * correct / total
        trial.report(accuracy, epoch)
        # print(f'Total accuracy: {accuracy} %')
        writer.add_scalar('epoch-accuracy', accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        # print accuracy for each class
        # for classname, correct_count in correct_pred.items():
        #     precision = 100 * float(correct_count) / total_pred[classname] if total_pred[classname] > 0 else 0
        #     recall = 100 * float(correct_count) / total_label[classname] if total_label[classname] > 0 else 0
        #     print(f'{classname:5s} Precision: {precision:.1f}, Recall: {recall:.1f}, Count Pred: {total_pred[classname]}, Count GT: {total_label[classname]}')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metric': accuracy
            }

        if accuracy > BEST:
            BEST = accuracy
            torch.save(checkpoint, "classification.pt")
    return accuracy


if __name__ == "__main__":
    pass
    # study = optuna.create_study(direction="maximize",
    #     pruner=optuna.pruners.PercentilePruner(
    #         25.0, n_startup_trials=5, n_warmup_steps=15, interval_steps=5
    #     ))
    # study.optimize(objective, n_trials=40)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # with open('opimization.txt',"w") as f:
    #     f.write("Study statistics:\n")
    #     f.write("  Number of finished trials:{}".format(len(study.trials)))
    #     f.write("  Number of pruned trials:{}".format(len(pruned_trials)))
    #     f.write("  Number of complete trials:{}".format(len(complete_trials)))

    #     f.write("Best trial:\n")
    #     trial = study.best_trial

    #     f.write("  Value:{}".format(trial.value))

    #     f.write("  Params: \n")
    #     for key, value in trial.params.items():
    #         f.write("    {}: {}".format(key, value))
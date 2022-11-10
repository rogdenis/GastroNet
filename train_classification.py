#https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A
from utils import ClassificationDataset, collate_fn
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
writer = SummaryWriter('classification')

image_transform = A.Compose([
    A.MotionBlur(p=0.5),
    A.Defocus(p=0.5)
])

coords_transform = A.Compose([
    A.Affine(p=0.5),
    A.Flip(p=0.5)
])

batch_size = 4

trainset = ClassificationDataset('classification/train', '_classes.csv',
    image_transform=image_transform,
    coords_transform=coords_transform,
)
print(trainset.__len__())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

valid = ClassificationDataset('classification/valid', '_classes.csv',
    image_transform=None,
    coords_transform=None,
)
validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size,
                                         shuffle=False)

classes = ('Antrum pyloricum', 'Antrum pyloricun', 'Corpus gastricum', 'Duodenum', 'Esophagus III/III', 'Mouth', 'Oesophagus', 'Oropharynx', 'Pars cardiaca')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(7, 16, 5)
        self.fc1 = nn.Linear(16 * 101 * 101, 120)
        self.fc2 = nn.Linear(120, 20)
        self.fc3 = nn.Linear(20, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LR = 0.0001
WD = 0.0001
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = resnet50()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay = WD)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
BEST = -1
for epoch in range(100):  # loop over the dataset multiple times
    print("Train epoch {}".format(epoch))
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        images.to(device)
        labels.to(device)
        #print(inputs.size())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        writer.add_scalar('training-loss', loss, epoch * len(trainloader) + i)
        if i % 100 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 2000:.5f}')
            running_loss = 0.0

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_label = {classname: 0 for classname in classes}
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
                    correct_pred[classes[label]] += 1
                    correct += 1
                total += 1
                total_pred[classes[prediction]] += 1
                total_label[classes[label]] += 1

    lr_scheduler.step()

    accuracy = 100 * correct / total
    print(f'Total accuracy: {accuracy} %')
    writer.add_scalar('epoch-accuracy', accuracy, epoch)
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        precision = 100 * float(correct_count) / total_pred[classname] if total_pred[classname] > 0 else 0
        recall = 100 * float(correct_count) / total_label[classname] if total_label[classname] > 0 else 0
        print(f'{classname:5s} Precision: {precision:.1f}, Recall: {recall:.1f}, Count Pred: {total_pred[classname]}, Count GT: {total_label[classname]}')

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
print('Finished Training')
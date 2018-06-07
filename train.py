from skimage import io
import torch
import os
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    """Face dataset."""

    def __init__(self, malePath, femalePath):
        self.male = [malePath+'/'+x for x in os.listdir(malePath)]
        self.female = [femalePath+'/'+x for x in os.listdir(femalePath)]

    def __len__(self):
        return len(self.male) + len(self.female)

    def __getitem__(self, idx):
        if idx < len(self.male):
            sample = {'image': io.imread(self.male[idx]),
                      'label': np.array([0.0])}
        else:
            sample = {'image': io.imread(self.female[idx-len(self.male)]),
                      'label': np.array([1.0])}
        return sample['image'], sample['label']


batch_size = 128
C, H, W = 3, 64, 64
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

model = torch.nn.Sequential(
          torch.nn.Conv2d(3, 8, 3, padding=1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2),
          torch.nn.Conv2d(8, 16, 3, padding=1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(2),
          Flatten(),
          torch.nn.Linear(16*16*16, 1)
        ).to(device)


criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

trainset = FaceDataset('./data_resized/Celeb_gender/trainA', './data_resized/Celeb_gender/trainB')
testset = FaceDataset('./data_resized/Celeb_gender/testA', './data_resized/Celeb_gender/testB')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)

epochs = 10

print("Starting training")


def find_accuracy(loader):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in loader:
                images, labels = data
                images = images.type('torch.FloatTensor').to(device)
                labels = labels.type('torch.LongTensor').to(device)
                images.transpose_(1, 3)
                images.transpose_(2, 3)
                outputs = model(images)
                predicted = (1+torch.sign(outputs))/2.0
                predicted = predicted.type('torch.LongTensor').to(device).data
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

    return correct/total


for e in range(epochs):
    i = 0
    for data in trainloader:
        #print(i)
        # get the inputs
        inputs, labels = data
        inputs = inputs.type('torch.FloatTensor').to(device)
        labels = labels.type('torch.FloatTensor').to(device)
        inputs.transpose_(1, 3)
        inputs.transpose_(2, 3)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        if i % 100 == 0:
            print("Epoch: ", e, "Iter: ", i, "Loss: ", loss)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            torch.save(model.state_dict(), 'saved_models/'+str(e)+'_'+str(i)+'.pt')
            print("Accuracy: ", find_accuracy(testloader))
        i += 1


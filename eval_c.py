# troch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import time
import torchvision.datasets as datasets
import torchvision.models as models

def eval(model, data_loader, device):
    print('Start test..')
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = model(inputs)

            loss += criterion(outputs, targets).item()

            # _, predicted = outputs.max(1)

            _, predicted = torch.max(outputs,1)
            # correct += predicted.eq(targets).sum().item()
            correct += (targets == predicted).sum().item()
        print('\nTotal average test acc : ', correct / total)
        print('total average test_loss :', loss / total)

# batch size
batch_size=24

test_path='C:/Users/User/AI/ch/q1/test/'
#test_path='C:/Users/User/AI/ch/q1/t_v_re2/val'

test_dataset = datasets.ImageFolder(
    test_path,
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]))

# loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# loss
criterion = nn.CrossEntropyLoss().to(device)

model = models.resnet50()
model = torch.nn.DataParallel(model).cuda()
model_load_path = "C:/Users/User/AI/ch/q1/hyeon_model_best.pth.tar"
checkpoint=torch.load(model_load_path)
model.load_state_dict(checkpoint['state_dict'])



eval(model, test_loader, device)


#
# model_load_path = "./weights/SimpleCNN/best_model.pt"
# model = SimpleCNN().to(device)
# model.load_state_dict(torch.load(model_load_path))

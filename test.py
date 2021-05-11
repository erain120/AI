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

            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()

        print('\nTotal average test acc : ', correct / total)
        print('total average test_loss :', loss / total)

        # save model

        state = {
            'net': model.state_dict()
        }

# batch size
batch_size=128

# dataset 구축
# args='C:/Users/User/AI/ch/q1/t_v_re2'
# traindir = os.path.join(args, 'train')
# valdir = os.path.join(args, 'val')
train_path='C:/Users/User/AI/ch/q1/t_v_re2/train/'
test_path='C:/Users/User/AI/ch/q1/1_test/'

train_dataset = datasets.ImageFolder(
    train_path,
    transforms.Compose([
        transforms.ToTensor()
    ]))

test_dataset = datasets.ImageFolder(
    test_path,
    transforms.Compose([
        transforms.ToTensor()
    ]))

# loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#net = net.to(device)

# loss
criterion = nn.CrossEntropyLoss().to(device)




#checkpoint = torch.load("C:/Users/User/AI/ch/q1")
#net.load_state_dict(checkpoint["hyeon_model_best.pth"])

model=models.resnet50()
#model = model.to(device)
model = torch.nn.DataParallel(model).cuda()
# opt
lr_val = 0.1
optimizer = optim.SGD(model.parameters(),lr=lr_val,momentum=0.9,weight_decay=0.0002)
model_load_path = "C:/Users/User/AI/ch/q1/model_best.pth.tar"
# model_load_path = "C:/Users/User/AI/examples/imagenet/model_best.pth.tar"
checkpoint=torch.load(model_load_path)
model.load_state_dict(checkpoint['state_dict'])


eval(model,test_loader,device)


#
# model_load_path = "./weights/SimpleCNN/best_model.pt"
# model = SimpleCNN().to(device)
# model.load_state_dict(torch.load(model_load_path))
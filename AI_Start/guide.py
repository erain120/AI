# [TODO] pytorch version check, GPU 사용 유무 체크, device 설정 코드
# pytorch version check, GPU 사용 유무 체크
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from eval import eval
from matplotlib import pyplot as plt

from utils.dataset import CustomDataset
from utils.transforms import data_transforms
from torch.utils.data import DataLoader
from model.vgg import * #vgg19 #vgg19에 대한 모델을 사용하기 위해서 inport


#graph그리기 위한 import
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from functools import partial
from threading import Thread
from tornado import gen


# # [TODO] Hyper parameter setting 코드 구현
# batch_size = None
# num_epochs = None
# lerning_rate = None
#
# # [TODO] custom dataset 코드 구현
# # Step 1 : 변환할 이미지 경로 정의 및 이미지 로드
# # 1. 인자로 받은 index 변수 (integer)를 이용하여 self.all_data 리스트에서 그 인덱스에 해당하는 데이터 경로를 정의 data_path 변수에 저장
# # 2. PIL 패키지의 Image.open 또는 CV2 cv2.imread() 을 통해 위에서 지정한 data_path 에 있는 PIL, CV2, 이미지 객체로 변환하고,
# #    그 결과를 img 라는 이름의 변수에 저장
# # 3. 만일 self.transform 에 정의된 것이 있다면 즉 None 이 아니라 img 에 self.transform 을 통해 데이터 전처리를 적용하세요
# # Step 2 : 이미지에 대한 라벨(label) 정의
# # 마지막으로 데이터의 라벨(label)을 정의할 차례입니다. 만약 데이터 경로인 data_path 가 고양이 이미지라면 label 변수에는 0을 강아지라면 1을 정의하세요. 데이터 파일의 파일 이름을 통해 이미지가 고양이인지 강아지인지 알 수 있습니다. (팁 : 사람에 따라서 다양한 방식으로 구현할 수 있겠지만,
# # os.path.basename 함수를 이용하면 파일 경로 문자열에서 파일 이름(+확장자)만 분리할 수 있습니다. 그리고 str.startwith을 활용하면 파일명 'cat' 으로
# # 시작하는지 혹은 "dog" 으로 시작하는지 알 수 있습니다.
#
# # 다음을 읽고 __len__ 함수를 완성해보세요.
# # 이 함수는 매우 간단합니다. 정의할 데이터 총 데이터 개수를 length 변에 저장하세요.
# from torch.utils.data import Dataset
#
# class CustomDataset(Dataset) :
#     def __init__(self, data_dir, mode, transform=None):
#         # 데이터 정의
#         self.all_data = None
#         self.transform = None
#
#     def __getitem__(self, item):
#         # Step 1 변환할 이미지 경로 정의 및 이미지 로드
#
#         # data_path = None # 위의 설명 Step 1의 1. 을 참고하여 None을 채우세요.
#         # img = None # 위의 설명 Step 1의 2. 을 참고하여 None을 채우세요.
#         # None # # 위의 설명 Step 1의 3. 을 참고하여 None을 채우세요.
#         # Step 2 : 이미지에 대한 label 정의
#         # label = None
#
#         # data_path = None # 위의 설명 Step 1의 1. 을 참고하여 None을 채우세요.
#         # img = None # 위의 설명 Step 1의 2. 을 참고하여 None을 채우세요.
#         # None # # 위의 설명 Step 1의 3. 을 참고하여 None을 채우세요.
#
#         data_path = None
#         img = None
#
#         label = None
#
#         return img, label
#
#     def __len__(self):
#
#         length = None
#
#         return length
#
# # [TODO] Data Augmentation 코드 구현
# import torchvision.transforms as transforms
#
# data_transforms = {
#     'train': transforms.Compose([
#
#     ]),
#     'val': transforms.Compose([
#
#     ]),
# }
#
# # [TODO] 데이터 정의 및 데이터 loader 코드 구현
# # data_dir = data path 경로 , shuffle = True and False
# from torch.utils.data import DataLoader
#
# train_data = CustomDataset(data_dir=None, mode='train', transform=data_transforms['train'])
# val_data = CustomDataset(data_dir=None, mode='val', transform=data_transforms['val'])
# test_data = CustomDataset(data_dir=None, mode='test', transform=data_transforms['val'])
#
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=None, drop_last=True)
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=None, drop_last=True)
# test_loader = DataLoader(test_data, batch_size=1, shuffle=None, drop_last=True)
#
#
# # [TODO] 모델 설정
# # num_classes = 라벨 갯수, 학습할 모델 명
# """
# alexnet
# resnet
# vgg
# squeezenet
# inception
# densenet
# """
# import torchvision.models as models
# #net = models.__dict__["모델명"](pretrained=False, num_classes=None).to(device) #torch에 있는 모델을 가져와서 쓰겠다.
# net = models.__dict__["vgg19"](pretrained=False, num_classes=None).to(device)
#
# # [TODO] 코드 구현 : train 함수
# # 1. 모델에 입력 이미지 를 전달하고 출력 결과를 outputs에 저장합니다.
# # 2. criterion 은 손실함수를 담은 객체입니다. 예측 값인 outputs 와 라벨 값인 labels 통해 손실값을 계산하고 그 결과를 loss 변수에 저장
# # 3. optimizer 은 옵티마이저입니다. 이전에 계산한 기울기를 모두 clear 하고, 오차 역전파(backpropagation)를 통해 기울기를 계산하고,
# # 4. lr_scheduler = 본인들이 원하는 스케줄러를 사용 예) step_size = 3 단계 gamma = 0.1
# # 옵티마이저를 통해 파라미터를 업데이트 합니다.
# import torch.nn as nn
# import torch.optim as optim
# criterion = None
# optimizer = None
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
#
# # [TODO] 코드 구현 : train 함수
# # 1. 모델에 입력 이미지 를 전달하고 출력 결과를 outputs에 저장합니다.
# # 2. criterion 은 손실함수를 담은 객체입니다. 예측 값인 outputs 와 라벨 값인 labels 통해 손실값을 계산하고 그 결과를 loss 변수에 저장
# # 3. optimizer 은 옵티마이저입니다. 이전에 계산한 기울기를 모두 clear 하고, 오차 역전파(backpropagation)를 통해 기울기를 계산하고,
# # 옵티마이저를 통해 파라미터를 업데이트 합니다.
#
# # 일정한 에폭마다 다음에 구현할 validation 함수를 통해 검증을 수행합니다. 모델 검증을 수행했을 때, 만약 검증 과정의 평균 loss가 현재까지 가장 낮다면 가장 잘 훈련된 모델로 가정하고 그때까지 학습한 모델을 저장합니다. 저장은 추후에 구현할 save_model 함수가 수행합니다
#
# def train(num_epochs, model, data_loader, criterion, optimizer, saved_dir, val_every, device) :
#
#     print("Start training .... ")
#     best_loss = 99999
#
#     for epoch in range(num_epochs) :
#         for i , (imgs, labels) in enumerate(data_loader) :
#
#             # 코드 시작 #
#             images , labels= None, None
#
#             outputs = None
#             loss = None
#
#             # Backward and optimize # Parameters 업데이트 : 위의 설명 3. 을 참고하여 작성 하세요.
#
#             # 코드 종 #
#             _, argmax = torch.max(outputs, 1)
#             accuracy = (labels == argmax).float().mean()
#
#             if (i + 1) % 3 == 0:
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
#                     epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))
#
#
#         if (epoch + 1) % val_every == 0:
#             avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
#             if avrg_loss < best_loss:
#                 print('Best performance at epoch: {}'.format(epoch + 1))
#                 print('Save model in', saved_dir)
#                 best_loss = avrg_loss
#                 save_model(model, saved_dir)
#
# # [TODO] 코드 구현 : validation 함수
# """
# validation 함수입니다. 다음을 읽고 코드를 완성해보세요.
#
# validation 과정에서는 파라미터 업데이트를 하지 않기 때문에 기울기를 계산할 필요는 없습니다. 하지만 validation 과정에서의 평균 loss를 계산하기 위해 loss는 계산해야 합니다.
# train 함수와 마찬가지로 model에 입력 이미지를 전달하여 얻은 출력 결과를 outputs에 저장하고, criterion을 통해 loss를 계산한 뒤, 그 결과를 loss에 저장합니다.
# """
# def validation(epoch, model, data_loader, criterion, device):
#     print('Start validation #{}'.format(epoch) )
#     model.eval()
#     with torch.no_grad():
#         total = 0
#         correct = 0
#         total_loss = 0
#         cnt = 0
#         for i, (imgs, labels) in enumerate(data_loader):
#             ## 코드 시작 ##
#
#             imgs, labels = None, None
#             outputs = None
#             loss = None
#
#             # ## 코드 종료 ##
#             total += imgs.size(0)
#             _, argmax = torch.max(outputs, 1)
#             correct += (labels == argmax).sum().item()
#             total_loss += loss
#             cnt += 1
#         avrg_loss = total_loss / cnt
#         print('Validation #{}  Accuracy: {:.2f}%  Average Loss: {:.4f}'.format(epoch, correct / total * 100, avrg_loss))
#     model.train()
#     return avrg_loss
#
# # [TODO] 코드 구현 : save_model 함수
# """
#  - torch.save를 통해 output_path 경로에 check_point 를 저장하세요.
# """
# import os
# def save_model(model, saved_dir, file_name='best_model.pt'):
#     # 학습 모델 저장 폴더 만들기
#     # saved_dir 변수 : 모델 저장 경로
#     ## 코드 시작 ##
#     # os.makedirs()이용
#
#     output_path = os.path.join(saved_dir, file_name)
#     # torch.save 이용하여 모델 저장
#     ## 코드 종료 ##
#
#
# # [TODO] 코드 구현 : test 함수
# """
# - model 에 입력 이미지를 전달하여 얻은 출력 결과를 outputs에 저장합니다
# """
# def test(model, data_loader, device):
#     print('Start test..')
#     # model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#
#         for i, (imgs, labels) in enumerate(data_loader):
#
#             ## 코드 시작 ##
#             imgs, labels = None,None
#             outputs = None
#             ## 코드 종료 ##
#
#             _, argmax = torch.max(outputs, 1)    # max()를 통해 최종 출력이 가장 높은 class 선택
#             total += imgs.size(0)
#             correct += (labels == argmax).sum().item()
#
#         print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))
#     model.train()
#
# # [TODO] Training
# """
# - model : 네트워크 모델 변수에 저장
# - val_every : 검증을 몇 epoch 마다 진행할지 정하는 변수
# - saved_dir = model weight 저장 위치
# """
# model = None
# val_every = None
# # save 모델 weights 저장 폴더 생성
# saved_dir = "모델 저장 위치"
#
# # train(num_epochs, model, train_loader, criterion, optimizer, saved_dir, val_every, device)
#
# # [TODO] 학습된 모델 테스트 eval()
# # model = 모델 선언
# # 학습한 모델 불러오기 load_state_dict() / torch.load() 함수 이용하여 만들기
# model_load_path = "학습한 모델 저장 위치 경로"
# model = None # 모델 선언
# model.load_state_dict(torch.load())
#
# # 실행하고 싶은경우 트레인 호출 부분을 주석 처리하고 실행 또는 eval.py 를 생성하여 작성 후 실행 하기
# eval(model, test_loader, device)
#
# def eval(model, data_loader, device):
#     print('Start test..')
#     model.eval()
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#
#         for i, (imgs, labels) in enumerate(data_loader):
#             ## 코드 시작 ##
#
#             imgs, labels = None, None
#             outputs = None
#
#             ## 코드 종료 ##
#             _, argmax = torch.max(outputs, 1)    # max()를 통해 최종 출력이 가장 높은 class 선택
#             total += imgs.size(0)
#             correct += (labels == argmax).sum().item()
#
#             # 후처리 코드 작성 필요 # 본인들이 보여주고 싶은 후처리 방식으로 코드 작성
#
#         print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))
#         print('End test..')

# [TODO] 코드 구현 : save_model 함수
# """
#  - torch.save를 통해 output_path 경로에 check_point 를 저장하세요.
# """

def save_model(model, saved_dir, file_name='best_model.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    # 학습 모델 저장 폴더 만들기
    # saved_dir 변수 : 모델 저장 경로
    ## 코드 시작 ##
    # os.makedirs()이용

    output_path = os.path.join(saved_dir, file_name)
    # torch.save 이용하여 모델 저장
    torch.save(model.state_dict(), output_path) #save할 모델, 경로
    ## 코드 종료 ##


#[TODO] 코드 구현 : validation 함수
"""
validation 함수입니다. 다음을 읽고 코드를 완성해보세요.

validation 과정에서는 파라미터 업데이트를 하지 않기 때문에 기울기를 계산할 필요는 없습니다. 하지만 validation 과정에서의 평균 loss를 계산하기 위해 loss는 계산해야 합니다.
train 함수와 마찬가지로 model에 입력 이미지를 전달하여 얻은 출력 결과를 outputs에 저장하고, criterion을 통해 loss를 계산한 뒤, 그 결과를 loss에 저장합니다.
"""
def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch) )
    va_list = []
    model.eval() #모델에 eval을 선언하면 학습하지 않고 비교를 진행함
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        loss_values = []

        for i, (images, labels) in enumerate(data_loader):
            ## 코드 시작 ##
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)


            # ## 코드 종료 ##
            total += images.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1

        avrg_loss = total_loss / cnt
        y_label=correct / total * 100
        #va_list.append(y_label)
        print('Validation #{}  Accuracy: {:.2f}%  Average Loss: {:.4f}'.format(epoch, y_label, avrg_loss))
    model.train()
    return avrg_loss, y_label

#[TODO] 코드 구현 : train 함수
def train(num_epochs, model, data_loader, criterion, optimizer, saved_dir, val_every, device) :

    print("Start training .... ")
    best_loss = 99999
    ta_list = []
    tlost_list = []
    vloss_list = []
    av_list = []

    for epoch in range(num_epochs) :
        for i , (imags, labels) in enumerate(data_loader) :
            #label_name
            #print("item값 : ", item)
            #print(item["label"])

            # 코드 시작 #

            # FIX
            # images, labels = item["image"].to(device), item["label"].to(device) #device에 들어가는 값은 int형태여야 함
            images = imags.to(device)
            labels = labels.to(device)
            # print("img : ", images)
            # print("la : ", labels)

            outputs = model(images) #model에 대한 결과물을 보기 위해 image값을 넣어 줌 // if)outputs가 존재X면, 결과물 볼 수 x
            loss = criterion(outputs, labels)

            optimizer.zero_grad() #기울기 초기화
            loss.backward()

            #optimizer갱신
            optimizer.step()

            # 코드 종 #
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax).float().mean()
            #plot 실시간으로 그리기

            at_label = accuracy.item()*100
            loss_label = loss.item()
            if (i + 1) % 3 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss_label, at_label))
                ta_list.append(at_label)
                tlost_list.append(loss_label)

            if (epoch + 1) % val_every == 0 :
                print("여기 되낭?", val_every, "             epoch : ", epoch + 1)
                avrg_loss, av_label = validation(epoch + 1, model, val_loader, criterion, device)
                av_list.append(av_label)
                if avrg_loss < best_loss:
                    print('Best performance at epoch: {}'.format(epoch + 1))
                    print('Save model in', saved_dir)
                    best_loss = avrg_loss
                    save_model(model, saved_dir)

    print("av : ", av_list)
    print("ta : ", ta_list)
    plt.plot(av_list,label='av')
    plt.plot(ta_list,label='ta')
    plt.show()


if __name__=='__main__' :

    # FIX


    print("pytorch version >>", torch.__version__)
    print("GPU 사용 가능 여부 >> ", torch.cuda.is_available())

    # device 설정 : GPU 사용 가능 여부에 따라 device 정보 저장 torch.cuda.is_available() 사용하여 작성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Available devices ', torch.cuda.device_count())

    #train set
    num_epochs = 100
    batch_size = 32

    #eval_set
    num_labels = 10  # label 개수

    #model set
    #net = models.__dict__["vgg19"](pretrained=False, num_classes=None).to(device) #num_classes 에 label의 갯수
    net = models.__dict__["mobilenet_v3_small"](pretrained=False, num_classes=10).to(device) #model = vgg19(pretrained=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                                weight_decay=0.0002)  # dampening=0, weight_decay=0, nesterov=False) default값으로 잡혀서 쓸 필요X
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    #data_loade set
    # train_data = CustomData를 생성해주고 train_loader는 data를 batch_size만큼 읽어주는 역할이라서 항상 쌍으로 존재해야 함 so, train원형에 학습할 데이터 train_loader를 넣어 줌
    train_data = CustomDataset(data_dir="C:/Users/User/Desktop/AI/AI_Srart/dataset/train", mode='train', transform=data_transforms['train'])
    val_data = CustomDataset(data_dir="C:/Users/User/Desktop/AI/AI_Srart/dataset/val", mode='val', transform=data_transforms['val'])
    test_data = CustomDataset(data_dir="C:/Users/User/Desktop/AI/AI_Srart/dataset/test", mode='test', transform=data_transforms['val'])


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) #from torch.utils.data 안에 DataLoader 함수를 import
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

    saved_dir = "C:/Users/User/Desktop/AI/AI_Srart/weight"
    val_every = 10
    model = net
    # org  - >  train(num_epochs, net, data_loader, criterion, optimizer, saved_dir, val_every, device)
    #train(num_epochs, model, train_loader, criterion, optimizer, saved_dir, val_every, device)


    # 학습한 모델 불러오기 load_state_dict() / torch.load() 함수 이용하여 만들기
    model_load_path = "C:/Users/User/Desktop/AI/AI_Srart/weight/best_model.pt"  #"학습한 모델 저장 위치 경로"
    model.load_state_dict(torch.load(model_load_path))

    #load_model = 'C:/Users/User/AI/ch/q2/Q2_MIXUP_First_best_resnet50_1.pth'  # test or resume 일 경우 작성 , 경로임
    eval(model, test_loader, device)
    #eval(model, test_loader, device, criterion, model_load_path, num_labels)


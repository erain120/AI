import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np

#실행하고 싶은경우 트레인 호출 부분을 주석 처리하고 실행 또는 eval.py 를 생성하여 작성 후 실행 하기
#eval(model, test_loader, device)
from matplotlib import pyplot as plt

"""
#def eval(model, data_loader, device) :
def Test_eval(a_model,data_loader,a_device,a_criterion,a_path,a_num_labels):

#data_loader = test할 data
    print('Start test..')
    model.eval() #train이 실행되지 않도록 막음
    correct = 0
    total = 0
    t_num_label = np.zeros(shape=a_num_labels)
    num_label = np.zeros(shape=a_num_labels)

label_name = ["chongkong", "hanfeng", "yueyawan", "shuiban", "youban", "siban", "7_yiwu", "8_yahen",
              "9_zhehen", "10_yaozhed"]
    y_graph = []

    with torch.no_grad():

        for i, (images, labels) in enumerate(data_loader):
            ## 코드 시작 ##

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            ## 코드 종료 ##
            _, argmax = torch.max(outputs, 1)    # max()를 통해 최종 출력이 가장 높은 class 선택
            total += images.size(0)
            correct += (labels == argmax).sum().item()
            #label = 정답지
            predict = argmax
            #print(images) #images 를 cv2.read해서 사진으로 저장하기
            num_label[predict] += (labels == argmax).sum().item()
            cnt_label[predict] += 1

            x = np.arange(10)
            plt.figure(figsize=(10, 8)) #windows size
            plt.bar(x, y_graph, width=0.6, align="center", color='springgreen')
            plt.xticks(x, label_name, fontsize="10", rotation=45)
            plt.show()


            # print("argmax : ", argmax)
            # print("labels : ", labels)
            # print("sum ", (labels == argmax).sum())
            # print("item ", (labels == argmax).sum().item())

            # 후처리 코드 작성 필요 # 본인들이 보여주고 싶은 후처리 방식으로 코드 작성

        print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))
        print('End test..')
        print("cnt 값 : ", num_label)"""

def eval(model, data_loader, device) :
#label의 갯수가 바뀌어도 eval을 할 수 있도록 수정
    print('Start test..')
    #checkpoint = torch.load(path)
    #model.load_state_dict(checkpoint)
    model.eval()
    correct = 0
    total = 0
    loss = 0
    total_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #np.zeros(shape=num_labels)
    score_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #np.zeros(shape=num_labels)
    y_label = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # correct += (labels == argmax).sum().item()
            # # label = 정답지
            #predict = argmax

            _, predicted = torch.max(outputs, 1)
            #argmax = predicted
            #print("arg : ", argmax.item())
            #print("predict : ", predicted)
            correct += predicted.eq(targets).sum().item()
            total_label[targets] += 1
            score_label[predicted] += (targets == predicted).sum().item()

            #t_num_label[targets] += 1
            #num_label[predicted] += (targets == predicted).sum().item()

        print("total_label : ", total_label)
        print("score_label : ", score_label)

        length = len(total_label)
        print("length :", length)

        for i in range(10):
            y_value = score_label[i]/total_label[i]*100
            print('{} label Accuracy of {:.2f} '.format(i, y_value))
            y_label.append(y_value)



        label_name = ["chongkong", "hanfeng", "yueyawan", "shuiban", "youban", "siban", "7_yiwu", "8_yahen", "9_zhehen", "10_yaozhed"]
        x = np.arange(length)
        plt.figure(figsize=(10, 8))
        plt.bar(x, y_label, width=0.6, align="center")
        plt.xticks(x, label_name, fontsize="10", rotation=45)
        plt.show()


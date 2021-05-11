import cv2
import os
import torch
from torch import nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
import numpy as np
import torchvision

def make_dir(name):
    if not os.path.isdir(name):
        os.makedirs(name)
    else:
        pass



folder_names = ['1_chongkong/','2_hanfeng/','3_yueyawan/','4_shuiban/','5_youban/',
               '6_siban/','7_yiwu/','8_yahen/','9_zhehen/','10_yaozhed/']


train_dir='C:/Users/User/AI/ch/q1/t_v/val/'
#val_dir='C:/Users/User/AI/ch/q1/t_v/val/'

save_dir='C:/Users/User/AI/ch/q1/t_v_re2/val/'
count=0
for i in folder_names:
    file_list = os.listdir(train_dir+i[:-1])
    for j in file_list:
        img_name = train_dir + i + j
        img = cv2.imread(img_name)
        h,w,c=img.shape
        ref = max(h, w)
        left = right = int((ref - w) / 2)
        top = bottom = int((ref - h) / 2)
        img2 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        img2 = cv2.resize(img2, (224, 224))

        new_dir=f'{save_dir}{i}/{count}.jpg'
        cv2.imwrite(new_dir, img2)
        count+=1

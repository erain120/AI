import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
import cv2
import shutil
import random

def make_dir(name):
    if not os.path.isdir(name):
        os.makedirs(name)
    else:
        pass

def move_file(or_local, new_local_train, new_local_val):
    train_file = []
    files = os.listdir(or_local)
    make_dir(new_local_train[:-1])
    make_dir(new_local_val[:-1])
    # 옮길 퍼센트
    test = int(len(files)*0.1)
    ran = random.sample(files, test)
    # 10%에 해당되지않는 파일들 train_file list 에 추가
    for i in files:
        if i not in ran:
            train_file.append(i)

    for a in range(test):
        file_name = or_local + str(ran[a])
        val_file_name = new_local_val + str(ran[a])
        shutil.copyfile(file_name, val_file_name)

    for a in range(len(train_file)):
        file_name2 = or_local + str(train_file[a])
        train_file_name = new_local_train + str(train_file[a])
        shutil.copyfile(file_name2, train_file_name)

# 이미지 파일 서치
image_format=[".bmp",".jpg",".png",".tif",".tiff"]
def file_search(folder_path):
    img_root=[]
    for (path,dir,file) in os.walk(folder_path):
        for filename in file:
            ext=os.path.splitext(filename)[-1].lower()
            # lower : 대문자->소문자
            if ext in image_format:
                root = os.path.join(path,filename)
                img_root.append(root)

    return img_root

# dataset 구성
class MyCustomDataset(Dataset):
    def __init__(self,path,transforms=None):
        self.path=path
        self.transforms=transforms
        # print(transforms)
    def __getitem__(self,item):
        path=self.path[item]
        img = cv2.imread(path)

        if self.transforms is not None:
            img =self.transforms(img)

        return img

    def __len__(self):
        return len(self.path)

# aug
data_transforms =transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.RandomRotation(degrees=0.5),
        transforms.RandomAffine(degrees=0.5,scale=(.9, 1.1), shear=0),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

folder_names = ['1_chongkong/','2_hanfeng/','3_yueyawan/','4_shuiban/','5_youban/',
               '6_siban/','7_yiwu/','8_yahen/','9_zhehen/','10_yaozhed/']
origin_dir = 'C:/Users/User/AI/ch/q1/1_train/'

train_dir='C:/Users/User/AI/ch/q1/t_v/train/'
val_dir='C:/Users/User/AI/ch/q1/t_v/val/'
# move_file(원본 폴더, train 폴더, val 폴더)
# move_file(origin_dir,train_dir,val_dir)

for folder_name in folder_names:
    move_file(origin_dir+folder_name, train_dir + folder_name, val_dir + folder_name)
    #train_data_path=os.listdir(train_dir + folder_name)
    train_data_path = file_search(train_dir + folder_name[:-1])
    dataset=MyCustomDataset(train_data_path,transforms=data_transforms)
    img_num = 1

    for _ in range(4):
        for img in dataset:
            save_image(img, train_dir + folder_name + str(img_num)+'.jpg')
            img_num += 1
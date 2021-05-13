import os
#from utils import search_file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import natsort

#dir_path = "C:/Users/User/Desktop/AI/AI_Srart/data/train/"
dir_path = ["C:/Users/User/Desktop/AI/AI_Srart/data/train/1_chongkong", "C:/Users/User/Desktop/AI/AI_Srart/data/train/2_hanfeng", "C:/Users/User/Desktop/AI/AI_Srart/data/train/3_yueyawan",
            "C:/Users/User/Desktop/AI/AI_Srart/data/train/4_shuiban", "C:/Users/User/Desktop/AI/AI_Srart/data/train/5_youban", "C:/Users/User/Desktop/AI/AI_Srart/data/train/6_siban",
            "C:/Users/User/Desktop/AI/AI_Srart/data/train/7_yiwu", "C:/Users/User/Desktop/AI/AI_Srart/data/train/8_yahen", "C:/Users/User/Desktop/AI/AI_Srart/data/train/9_zhehen",
            "C:/Users/User/Desktop/AI/AI_Srart/data/train/10_yaozhed"]
train_path = "C:/Users/User/Desktop/AI/AI_Srart/dataset/train"
val_path = "C:/Users/User/Desktop/AI/AI_Srart/dataset/val"

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# 이미지 파일 서치
image_format=[".bmp", ".jpg", ".png", ".tif", ".tiff"]
def search_file(folder_path):
    file_list=[]
    files = os.listdir(folder_path)
    # for _, i in enumerate(files) :
    #     a = os.path.join(dir_path, i)
    #     file_list.append(a)

    for (path, dir, file) in os.walk(folder_path):
        if dir is not None:
            file_path = dir
            for i in dir :
                file_list[i].append()

        for filename in file:
            #print(filename)
            ext=os.path.splitext(filename)[-1].lower()
            # lower : 대문자->소문자
            if ext in image_format:
                root = os.path.join(path, filename)
                file_list.append(root)
    return file_list

#LoadImage = f.open 비슷한느낌
class LoadImage :
    def __init__(self, path) :
        dir = str(Path(path)) #super path = path 경로 찾아주는 것
        dir_path = os.path.abspath(dir) # absolute path
        self.files = natsort.natsorted(search_file(dir_path))

        # print(""""---------------""")
        # print(self.files)

        self.img_number = len(self.files)
        # 23973
        print("img_number test : ", self.img_number)

    def __getitem__(self, item):
        path = self.files[item % len(self.files)]
        return path

    def __len__(self):
        return len(self.files)

for index, i in enumerate(dir_path) :
    dataset = LoadImage(dir_path[index])
    #print(" ###### : " , dataset[index])
    train_img_list, val_img_list = train_test_split(dataset, test_size=0.2, shuffle=False) #train_list, val_list를 test_size=0.5이면 50:50비율로 나누어 준다.
    # train_test_split 의 return 형태
    # return list(chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays))
    # print("train : ", train_img_list)
    # print("val_img_ : :", val_img_list)

    for i in range(0, len(train_img_list)):
        bas = os.path.basename(train_img_list[i])
        image = cv2.imread(train_img_list[i])
        name = bas.replace(".jpg", "")
        #print(train_img_list)
        file_str = train_img_list[i].split('\\')[-2:-1]
        file_str[0]
        os.makedirs(f"{train_path}/{file_str[0]}", exist_ok=True)
        cv2.imwrite(f"{train_path}/{file_str[0]}/{name}.png", image)

    for i in range(0, len(val_img_list)):
        bas = os.path.basename(val_img_list[i])
        image = cv2.imread(val_img_list[i])
        name = bas.replace(".jpg", "")
        file_str = val_img_list[i].split('\\')[-2:-1]
        file_str[0]
        os.makedirs(f"{val_path}/{file_str[0]}", exist_ok=True)
        cv2.imwrite(f"{val_path}/{file_str[0]}/{name}.png", image)

    # name = bas.replace(".png" , "")
    # img = cv2.imread(train_img_list[i])
    # cv2.imwrite(f"./drug_1/{name}.png", img)
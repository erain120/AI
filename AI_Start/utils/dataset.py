# [TODO] custom dataset 코드 구현
# Step 1 : 변환할 이미지 경로 정의 및 이미지 로드
# 1. 인자로 받은 index 변수 (integer)를 이용하여 self.all_data 리스트에서 그 인덱스에 해당하는 데이터 경로를 정의 data_path 변수에 저장
# 2. PIL 패키지의 Image.open 또는 CV2 cv2.imread() 을 통해 위에서 지정한 data_path 에 있는 PIL, CV2, 이미지 객체로 변환하고,
#    그 결과를 img 라는 이름의 변수에 저장
# 3. 만일 self.transform 에 정의된 것이 있다면 즉 None 이 아니라 img 에 self.transform 을 통해 데이터 전처리를 적용하세요
# Step 2 : 이미지에 대한 라벨(label) 정의
# 마지막으로 데이터의 라벨(label)을 정의할 차례입니다. 만약 데이터 경로인 data_path 가 고양이 이미지라면 label 변수에는 0을 강아지라면 1을 정의하세요. 데이터 파일의 파일 이름을 통해 이미지가 고양이인지 강아지인지 알 수 있습니다. (팁 : 사람에 따라서 다양한 방식으로 구현할 수 있겠지만,
# os.path.basename 함수를 이용하면 파일 경로 문자열에서 파일 이름(+확장자)만 분리할 수 있습니다. 그리고 str.startwith을 활용하면 파일명 'cat' 으로
# 시작하는지 혹은 "dog" 으로 시작하는지 알 수 있습니다.

# 다음을 읽고 __len__ 함수를 완성해보세요.
# 이 함수는 매우 간단합니다. 정의할 데이터 총 데이터 개수를 length 변에 저장하세요.
import glob
import os

import cv2
from torch.utils.data import Dataset

class CustomDataset(Dataset) :
    #mode는 train인지 val인지 설정
    #transform = Aug먹인다
    def __init__(self, data_dir, mode, transform=None):
        # 데이터 정의  #c++ class로 치면 private, public 설정

        self.all_data = sorted(glob.glob(os.path.join(data_dir,  "*/*"))) #train과 val나눈 후 dataset안에 data인 *png불러옴
        self.transforms = transform

    def __getitem__(self, item):
        # Step 1 변환할 이미지 경로 정의 및 이미지 로드
        # data_path = None # 위의 설명 Step 1의 1. 을 참고하여 None을 채우세요.
        # img = None # 위의 설명 Step 1의 2. 을 참고하여 None을 채우세요.
        # None # # 위의 설명 Step 1의 3. 을 참고하여 None을 채우세요.
        # Step 2 : 이미지에 대한 label 정의
        # label = None

        # data_path = None # 위의 설명 Step 1의 1. 을 참고하여 None을 채우세요.
        # img = None # 위의 설명 Step 1의 2. 을 참고하여 None을 채우세요.
        # None # # 위의 설명 Step 1의 3. 을 참고하여 None을 채우세요.

        data_path = self.all_data[item]
        # self.labels = self.label(data_path)
        from PIL import Image
        image = Image.open(data_path).convert("RGB")

        if self.transforms is not None :
            image = self.transforms(image)


        label = os.path.split(data_path)[0]  #['1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu', '8_yahen', '9_zhehen', '10_yaozhed']
        label_name = os.path.split(label)[1]
        label_index = []
        #print(label_name)
        if label_name == '1_chongkong' :
            label = 0
        elif label_name == '2_hanfeng' :
            label = 1
        elif label_name == '3_yueyawan' :
            label = 2
        elif label_name == '4_shuiban' :
            label = 3
        elif label_name == '5_youban' :
            label = 4
        elif label_name == '6_siban' :
            label = 5
        elif label_name == '7_yiwu' :
            label = 6
        elif label_name == '8_yahen' :
            label = 7
        elif label_name == '9_zhehen' :
            label = 8
        elif label_name == '10_yaozhed' :
            label = 9


        return image, label

    def __len__(self):
        return len(self.all_data) #한줄한줄씩 가져오는거 X, 전체 data의 갯수

        # length = None
        # return length



# data_path = "C:/Users/User/Desktop/AI/AI_Srart/dataset/train"
# dataset = CustomDataset(data_path, mode='train')
#
# for i, item in enumerate(dataset) :
#     pass
    # image = item["image"]
    # # print("image : ", image)
    #
    # label = item["label"]
    # print("label : ", label) #label :  4_shuiban
    #
#     # print(label)

import os
import xml.etree.ElementTree as ET
import cv2
from PIL import Image

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        error_msg = str(e)
        return False


# ext. train img name
path_dir = "C:/Users/User/AI/ch/q1/1_train"
file_list = os.listdir(path_dir)

# xml읽어서 pb_name 따오기, crop 하기
label_path='C:/Users/User/AI/ch/q1/1_label'
#crop_size={}
for i in file_list:
    file_name=label_path + '/' + i[:-3]+'xml'
    if(os.path.isfile(file_name)):
        tree = ET.parse(file_name)
        root = tree.getroot()
        obj_tag = root.findall('object')
        # pb_name 따오기
        label_name = obj_tag[0].find('name').text
        # crop할 사이즈 따오기
        xmin = int(obj_tag[0].find('bndbox/xmin').text)
        ymin = int(obj_tag[0].find('bndbox/ymin').text)
        xmax = int(obj_tag[0].find('bndbox/xmax').text)
        ymax = int(obj_tag[0].find('bndbox/ymax').text)
        #crop_size[i]=[xmin,ymin,xmax,ymax]
        if not os.path.exists('C:/Users/User/AI/ch/q1/1_train/' + label_name):
            os.makedirs('C:/Users/User/AI/ch/q1/1_train/' + label_name)

        # crop
        img1 = Image.open(path_dir + '/' + i)
        # left up right down
        img2 = img1.crop((xmin,ymin,xmax,ymax)




                         )
        dir11=f'C:/Users/User/AI/ch/q1/1_train/{label_name}/{i}'
        img2.save(dir11)




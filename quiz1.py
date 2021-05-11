import numpy as np
import cv2
import os

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        return None


path_dir = 'C:/Users/User/AI/ch/q1/1_label/label'

file_list = os.listdir(path_dir)
print(file_list)

for i in file_list:
    img_name = path_dir + '/' + i
    print(img_name)
    f = open(img_name, 'r')
    s = f.read()
    print(s)
    f.close()
    cv2.waitKey()
exit()
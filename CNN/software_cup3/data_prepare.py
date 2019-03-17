import base64
import json
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

dir = "/home/wangheng/Downloads/资料下载/my_credit_card/"

image_file = [f for f in listdir(dir) if isfile(join(dir, f)) and not f.endswith("json")]

for file_name in image_file:
    file_path = dir + file_name
    print(file_path)
    imread = cv2.imread(file_path)
    resize = cv2.resize(imread, (408, 306))
    cv2.imwrite(file_path, resize)
    print(file_path, " : ", imread.shape)

# with open(dir+image_file[0].split(".")[0]+".json", "r") as f:
#     print(image_file[0], "===", image_file[0].split(".")[0]+".json")
#     my_dict = json.load(f)
# for key in my_dict:
#     print(key)



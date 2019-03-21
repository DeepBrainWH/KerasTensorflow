from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import math
import json
import read_config_utils

dir = read_config_utils.TEST_IMAGE_PATH
image_file = [f for f in listdir(dir) if isfile(join(dir, f)) and not f.endswith("json")]

BATCH_SIZE = 3
TRAIN_DATA = 9
BATCHES = int(math.ceil(TRAIN_DATA / BATCH_SIZE))


class GET_DATA:

    def __init__(self):
        pass

    @staticmethod
    def get_batches_data():
        position = 0
        X = []
        y = []
        for i in range(TRAIN_DATA):
            file_path = dir + image_file[position + i]
            X.append(cv2.imread(file_path))
            with open(file_path.split(".")[0] + ".json", 'r') as f:
                load = json.load(f)
            y.append([load['shapes'][0]["points"], load['shapes'][1]["points"]])
        X = np.asarray(X)
        X = (X-np.min(X))/(np.max(X)-np.min(X))
        print(X)
        y = np.asarray(y)
        wh = [306,]
        y = y.reshape(y.shape[0], -1)
        return X, y

# train_X, train_y = GET_DATA.get_batches_data()
# print(train_X, train_y)

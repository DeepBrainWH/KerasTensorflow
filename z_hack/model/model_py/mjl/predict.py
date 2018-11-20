#!/usr/app/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
    In this file, we will predict the value of 1th, 2014.
"""
from __future__ import absolute_import, division, print_function

import sys
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import load_model

class Predict:

    def __init__(self):
        self.model = None

    def predict(self, old_data, model_path=None):
        try:
            if self.model is None:
                self.model = load_model(model_path, custom_objects={'MinimalRNNCell':MinimalRNNCell})
            predict_data = self.model.predict(old_data)
            old_data = old_data.reshape(1, 20)[:, 0:19]
            next_old_data = np.append(old_data, predict_data).reshape(1, 20, 1)
            return predict_data, next_old_data
        except FileNotFoundError as f:
            print('Please input correct model\'s path!')
            sys.exit(0)

    def get_data(self, datafile=None):
        if datafile is None:
            datafile = '../../../data/all_part'
        try:
            dataframe = pd.read_csv(datafile, sep='\t')
            np_price = dataframe.iloc[:, -1].values
            np_price = (np_price - np_price.min()) / (np_price.max() - np_price.min())
            np_price = np_price[-20:]
            return np.asarray(np_price, dtype=np.float32).reshape(1, 20, 1)
        except:
            raise FileNotFoundError('Please input correct data file path!')



total = len(sys.argv)
if total < 1:
    print('please input the train model\'s path!\tusage: ./predict <your model path>')
    sys.exit(0)
else:
    y = []
    model_path = './model_fifth.h5'
    predict = Predict()
    old_data = predict.get_data()
    for i in range(22):
        predict_data, old_data = predict.predict(old_data, model_path)
        y.append(predict_data * 4920.613604 + 2872.819729)
    y = np.asarray(y, dtype=np.float32)
    print(y)
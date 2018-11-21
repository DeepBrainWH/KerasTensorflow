#!/usr/app/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
    In this file, we will predict the value of 1th, 2014.
"""
import sys
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import load_model
from model_2_RNN import MinimalRNNCell

class Predict:

    def __init__(self):
        self.model = None

    def predict(self, old_data, model_path=None):
        try:
            if self.model is None:
                self.model = load_model(model_path, custom_objects={'MinimalRNNCell':MinimalRNNCell})
            predict_data = self.model.predict(old_data)
            return predict_data
        except FileNotFoundError as f:
            print('Please input correct model\'s path!')
            sys.exit(0)

    def get_data(self, datafile=None):
        if datafile is None:
            datafile = '../../../data/all_part'
        try:
            x = []
            dataframe = pd.read_csv(datafile, sep='\t')
            np_price = dataframe.iloc[:, -1].values
            np_price = (np_price - np_price.min()) / (np_price.max() - np_price.min())
            for i in range(np_price.shape[0] // 20):
                x.append(np_price[i*20: i*20 + 20])
            return np.asarray(x, dtype=np.float32).reshape(-1, 20, 1)
        except:
            raise FileNotFoundError('Please input correct data file path!')

def plot_data(c_data, p_data):
    import matplotlib.pyplot as plt
    plt.plot(c_data, label='correct data')
    plt.plot(p_data, label='predict_data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    total = len(sys.argv)
    if total < 1:
        print('please input the train model\'s path!\tusage: ./predict <your model path>')
        sys.exit(0)
    else:
        predict_data = []
        model_path = './model_tensorboard_3.h5'
        predict = Predict()
        old_data = predict.get_data()
        for i in range(old_data.shape[0]):
            y = predict.predict(old_data[i].reshape(1, 20, 1), model_path) * 4920.613604 + 2872.819729
            predict_data.append(y[0, :])
        predict_data = np.asarray(predict_data, dtype=np.float32)
        predict_data_20 = predict_data[:, 0:20]
        old_data = old_data.reshape(-1) * 4920.613604 + 2872.819729
        predict_data_20 = np.append(old_data[0:20], predict_data_20.reshape(-1))
        plot_data(old_data, predict_data_20)
        print(predict_data[-1, :])
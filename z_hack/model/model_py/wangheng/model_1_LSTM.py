# -*- coding: utf-8 -*-
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, Dropout, Activation
from tensorflow.python.keras.models import  Sequential, load_model
from tensorflow.python.keras.layers.normalization import BatchNormalization
import datetime


class MyLSTM():

    def __init__(self):
        self.TIMESTEP = 20
        self.TRAININGEXAMPLE = None
        self.model = None
        self.data_frame = None
        self.original_data = None
        self.x_train = None
        self.y_train = None
        self.train_data_mean = None
        self.train_data_min = None
        self.train_data_max = None
        self.history = None

    def get_data(self, filepath=None):
        if filepath is None:
            filepath = "../../../data/all_part"
        self.data_frame = pd.read_csv(filepath, sep='\t')
        self.original_data = self.data_frame.iloc[:, -1].values
        self.train_data_max = self.original_data.max()
        self.train_data_min = self.original_data.min()
        self.train_data_mean = self.original_data.mean()
        self.TRAININGEXAMPLE = self.original_data.shape[0]
        x = []
        y = []
        for i in range(self.TRAININGEXAMPLE - self.TIMESTEP):
            x.append(self.original_data[i: i+self.TIMESTEP])
            y.append(self.original_data[i+self.TIMESTEP])

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # training data normalization
        x = (x - self.train_data_min) / (self.train_data_max - self.train_data_min)
        y = (y - self.train_data_min) / (self.train_data_max - self.train_data_min)
        self.x_train = np.reshape(x, newshape=[x.shape[0], x.shape[1], 1])
        self.y_train = np.reshape(y, newshape=[y.shape[0], 1])
        return self.x_train, self.y_train

    def buildModel(self, model_path=None):
        try:
            if model_path is None:
                model_path = './model_tensorboard_1.h5'
            mymodel = load_model(model_path)
            history = mymodel.fit(self.x_train, self.y_train, batch_size=50, epochs=500, verbose=0, validation_split=0.2, callbacks=[TensorBoard()])
            self.history = history.history
            mymodel.save('./model_tensorboard_1.h5')
            self.model = mymodel
            self._write_val_loss_to_csv()
        except:
            start = datetime.datetime.now()
            mymodel = Sequential()
            mymodel.add(CuDNNLSTM(50, input_shape=(20, 1), return_sequences=True))
            mymodel.add(Activation('sigmoid'))
            mymodel.add(BatchNormalization())
            mymodel.add(Dropout(0.2))

            mymodel.add(CuDNNLSTM(100, return_sequences=True))
            mymodel.add(Activation('sigmoid'))
            mymodel.add(BatchNormalization())
            mymodel.add(Dropout(0.2))

            mymodel.add(CuDNNLSTM(100))
            mymodel.add(Activation('tanh'))
            mymodel.add(BatchNormalization())
            mymodel.add(Dropout(0.2))

            mymodel.add(Dense(50, activation='sigmoid'))
            mymodel.add(BatchNormalization())
            mymodel.add(Dropout(0.2))

            mymodel.add(Dense(20, activation='sigmoid'))
            mymodel.add(BatchNormalization())
            mymodel.add(Dropout(0.2))

            mymodel.add(Dense(1, activation='linear'))

            mymodel.compile('adam', 'mae', metrics=['mae'])
            print(mymodel.summary())
            self.model = mymodel
            history = mymodel.fit(self.x_train, self.y_train, batch_size=50, epochs=3000, verbose=1, validation_split=0.2, callbacks=[TensorBoard()])
            self.history = history.history
            mymodel.save('./model_tensorboard_1.h5')
            end = datetime.datetime.now()
            print('耗时',end-start)
            self._write_val_loss_to_csv()

    def _write_val_loss_to_csv(self):
        val_loss = self.history['val_loss']
        val_loss = np.asarray(val_loss, dtype=np.float32)
        df = pd.DataFrame(val_loss)
        df.to_csv('./val_loss_1.csv', mode='a', header=False)
if __name__ == '__main__':
    mylstm = MyLSTM()
    x_train, y_train = mylstm.get_data()
    mylstm.buildModel()
    print("mode build successful!")


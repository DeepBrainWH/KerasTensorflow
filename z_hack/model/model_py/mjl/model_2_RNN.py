# -*- encode=utf-8 -*-
"""
    此模型是z-hack算法比赛的另外一个模型，使用RNN进行训练和预测
"""
from tensorflow.python.keras.layers import Dense, RNN, Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Input, activations
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.utils import multi_gpu_model
import pandas as pd
import numpy as np


class MinimalRNNCell(Layer):

    def __init__(self, units=32, activation="tanh", **kwargs):
        self.units = units
        self.state_size = units
        self.activation = activation
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        activation = activations.get(self.activation)
        output = activation(output)
        return output, [output]


class BuildModel():

    def __init__(self):
        self.TIMESTEP = 20
        self.DATA_DIM = 1
        self.model = None
        self.rnnLayer = None
        self.input = None
        self.output = None
        self.x_train = None
        self.y_train = None
        self.history = None
        self.filename = r"../../data/all_part"
        self.original_data = pd.read_csv(self.filename, sep="\t")

    def __get_data(self):
        x = []
        y = []
        data = self.original_data.iloc[:, -1].values
        for i in range(len(data) - self.TIMESTEP):
            x.append(data[i:i + self.TIMESTEP])
            y.append(data[i + self.TIMESTEP])
        x = np.asarray(x, dtype=np.float32)
        x = x / x.max()
        y = np.asarray(y, dtype=np.float32)
        y = y / y.max()
        self.x_train = x.reshape([x.shape[0], x.shape[1], 1])
        self.y_train = y.reshape([y.shape[0], 1])
        self.input = Input(shape=(20, 1), name="input_tensor")

    def __built_multi_cell_Layer(self):
        """
        :return: the output tensor.
        """
        o1 = RNN(MinimalRNNCell(32, "tanh"), return_sequences=True)(self.input)
        o2 = RNN(MinimalRNNCell(32, "tanh"), return_sequences=True)(o1)
        o3 = RNN(MinimalRNNCell(32, "tanh"), return_sequences=True)(o2)
        o4 = RNN(MinimalRNNCell(32, "tanh"), return_sequences=False)(o3)
        o5 = Dense(1, activation="tanh")(o4)
        self.output = o5

    def build_model(self):
        if self.input is None:
            self.__get_data()
        if self.rnnLayer is None:
            self.rnnLayer = self.__built_multi_cell_Layer()

        if self.model is None:
            try:
                self.model = load_model("model_sec.h5")
                parallel_model = multi_gpu_model(self.model, gpus=1)
                self.history = parallel_model.fit(self.x_train, self.y_train, 20, epochs=50, verbose=2)
                parallel_model.save("./model_sec.h5")
            except:
                if not isinstance(self.model, Sequential):
                    self.model = Model(inputs=self.input, outputs=self.output)
                    self.model.compile("adam", loss="mse", metrics=["mae"])
                    print(self.model.summary())
                    self.model.fit(self.x_train, self.y_train, 20, 50, 2, validation_split=0.2)
                    self.model.save("./model_sec.h5")


if __name__ == "__main__":
    # mymodel = BuildModel()
    # mymodel.build_model()
    mymodel = load_model(r"model_sec.h5",custom_objects={"MinimalRNNCell":MinimalRNNCell})

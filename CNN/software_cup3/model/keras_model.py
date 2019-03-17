from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.models import load_model
from get_data import GET_DATA
import matplotlib.pyplot as plt
import cv2
import numpy as np

def build_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), (1, 1), "SAME", activation="relu", input_shape=(306, 408, 3)))
    model.add(MaxPool2D((3, 3), (2, 2), 'same'))
    model.add(Conv2D(64, (5, 5), (1, 1), "SAME", activation="relu"))
    model.add(MaxPool2D((3, 3), (2, 2), 'same'))
    model.add(Conv2D(64, (5, 5), padding="SAME", activation='relu'))
    model.add(MaxPool2D((3, 3), (2, 2), 'same'))
    model.add(Conv2D(16, (5, 5), padding="SAME", activation='relu'))
    model.add(MaxPool2D((3, 3), (2, 2), 'same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(8, activation='relu'))
    optimizer = Adadelta()
    model.compile(optimizer, loss=mean_squared_error)
    print(model.summary())
    train_X, train_y = GET_DATA.get_batches_data()
    cost_values = []
    for step in range(1000):
        cost = model.train_on_batch(train_X, train_y)
        cost_values.append(cost)
        if step % 10 == 0:
            print("step %d , cost value is %.3f" % (step, cost))
    model.save("./model1.h5")
    plt.plot(cost_values)
    plt.show()

def load_my_model():

    model = load_model("./model4.h5")
    image = cv2.imread("/home/wangheng/Downloads/资料下载/my_credit_card/2.jpeg")
    cv2.imshow("hello", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    resize = np.reshape(image, [1, 306, 408, 3])
    result = model.predict(resize)
    print(result[0])
    dst = cv2.rectangle(image, (result[0, 0], result[0, 1]),(result[0, 2], result[0, 3]), (0, 0, 255), 5)
    dst = cv2.rectangle(dst, (result[0, 4], result[0, 5]), (result[0, 6], result[0, 7]), (255, 0, 0), 5)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def retrain_model():
    model = load_model("./model3.h5")
    train_X, train_y = GET_DATA.get_batches_data()
    cost_values = []
    for step in range(100):
        cost = model.train_on_batch(train_X, train_y)
        cost_values.append(cost)
        if step % 10 == 0:
            print("step %d , cost value is %.3f" % (step, cost))
    model.save("./model4.h5")
    plt.plot(cost_values)
    plt.show()


# build_model()
load_my_model()
# retrain_model()
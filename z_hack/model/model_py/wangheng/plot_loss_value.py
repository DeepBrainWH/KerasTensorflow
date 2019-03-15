# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

df2 = pd.read_csv("./wangheng/val_loss_2.csv")
val_loss_2 = df2.iloc[:, -1].values
plt.figure(figsize=(20, 10))
plt.plot(val_loss_2)
plt.title('Model accuracy')
plt.ylabel('abs_loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
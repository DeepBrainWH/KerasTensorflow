
# coding: utf-8

# ### 1. 进行数据预处理

# In[1]:


from keras.utils import np_utils
import numpy as np
np.random.seed(10)


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')


# In[4]:



# ## 数据标准化，提高模型预测的准确度，并且加快收敛素的

# In[5]:


x_train = x_train/255
x_test = x_test/255


# In[6]:


y_trian_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)


# In[7]:



# #### 建立模型

# In[8]:


from keras.models import Sequential
from keras.layers import Dense


# In[9]:


model = Sequential()


# In[10]:


model.add(Dense(units=256,
               input_dim=784,
               kernel_initializer='normal',
               activation='relu'))

model.add(Dense(units=10,
               kernel_initializer='normal',
               activation='softmax'))


# In[11]:


print(model.summary())


# In[12]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam', metrics=['accuracy'])


# In[13]:


train_history = model.fit(x=x_train, y=y_trian_onehot,
                         validation_split=0.2,
                         epochs=20, batch_size=200, verbose=2)


# ### 显示训练过程

# In[14]:


import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# In[15]:
show_train_history(train_history, 'acc', 'val_acc')


# ### 用测试数据评估模型准确路

# In[16]:

scores = model.evaluate(x_test, y_test_onehot)
print('accuracy = ', scores[1])

# ### 进行预测

# In[17]:


prediction = model.predict_classes(x_test)


# In[18]:



# In[19]:


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = 'label=' + str(labels[idx])
        if len(prediction) > 0:
            title += ", predict = " + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()


# In[20]:


x_test_image = x_test.reshape(10000, 28, 28)
plot_images_labels_prediction(x_test_image, y_test, prediction, idx=340)


# ### describing confusion matrix / error matrix
# If we want to know which handwritten-digital has a higher accuracy, the confusion matrix would help us to display them

# In[24]:


import pandas as pd
pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict'])


# In[31]:


df = pd.DataFrame({'label': y_test, 'predict': prediction})



# In[33]:




# In[34]:


plot_images_labels_prediction(x_test_image, y_test, prediction, idx=340, num=1)


# In[35]:


plot_images_labels_prediction(x_test_image, y_test, prediction, idx=4271, num=1)

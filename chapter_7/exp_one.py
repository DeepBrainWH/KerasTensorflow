
# coding: utf-8

# ### 1. 进行数据预处理

# In[1]:


from keras.utils import np_utils
import numpy as np
np.random.seed(10)


# In[2]:


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')


# In[8]:


x_train[0]


# ## 数据标准化，提高模型预测的准确度，并且加快收敛素的

# In[9]:


x_train = x_train/255
x_test = x_test/255


# In[10]:


y_trian_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)


# In[11]:


y_trian_onehot[0]


# #### 建立模型

# In[12]:


from keras.models import Sequential
from keras.layers import Dense


# In[13]:


model = Sequential()


# In[14]:


model.add(Dense(units=256,
               input_dim=784,
               kernel_initializer='normal',
               activation='relu'))

model.add(Dense(units=10,
               kernel_initializer='normal',
               activation='softmax'))


# In[15]:


print(model.summary())


# In[16]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_history = model.fit(x=x_train, y=y_trian_onehot,
                         validation_split=0.2,
                         epochs=20, batch_size=200, verbose=2)


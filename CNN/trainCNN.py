import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Train:
    def __init__(self):
        pass

    def weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape, name='W'))

    def bias(self, shape):
        return tf.Variable(tf.truncated_normal(shape, name='B'))

    def conv2d(self, input, filter, output, padding=None, stride=None):
        if stride is None:
            stride = [1, 1, 1, 1]
        if padding is None:
            padding = 'SAME'
        return tf.nn.conv2d(input, filter, stride, padding, name='conv')

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')





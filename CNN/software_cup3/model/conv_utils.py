import tensorflow as tf


def conv(X, kernel_shape, stride, activation='relu', use_bias=True, padding='SAME', device='/GPU:0',
         with_pooling=False, pooling_size=None, pooling_stride=None, pooling_padding='SAME', name='ConvNet'):
    """
    :param X: input tensor: 4-dim: [batch_size, height, width, channel]
    :param kernel_shape: filter kernel: 4-dim: [x, y, in-dim, out-dim]
    :param stride: 2-dim: [x, y]
    :param activation: activation function
    :param use_bias: add bias
    :param padding: use padding: SAME or VALID
    :param name: layers name
    :param with_pooling: default is max_pool.
    :param pooling_size: 2-dim. if with_pooling is true, you have to assign the pooling size.
    :param pooling_stride:  2-dim. if with_pooling is true, you have to assign the pooling stride.
    :param pooling_padding: default value is 'SAME', you can also use 'VALID'.
    :return: convolution result and result shape.
    """
    with tf.name_scope(name):
        with tf.device(device):
            kernel = tf.random.normal(kernel_shape, name='kernel')
            conv_ = tf.nn.conv2d(X, kernel, [1, stride[0], stride[1], 1], padding, name='conv_layer')
            if use_bias:
                bias_ = tf.random.normal([kernel_shape[3]], name='bias')
                conv_ = tf.nn.bias_add(conv_, bias_, name='bias_add')
            if activation == 'relu':
                conv_ = tf.nn.relu(conv_, activation)
            elif activation == 'relu6':
                conv_ = tf.nn.relu6(conv_, activation)
            elif activation == 'sigmoid':
                conv_ = tf.nn.sigmoid(conv_, activation)
            elif activation == 'tanh':
                conv_ = tf.nn.tanh(conv_, activation)

            if with_pooling:
                if pooling_size is None or pooling_stride is None:
                    raise Exception("If using pooling, you have to assign the value of pooling_size and pooling_stride")
                else:
                    conv_ = tf.nn.max_pool(conv_, pooling_size, [1, pooling_stride[0], pooling_stride[1], 1],
                                           pooling_padding, name='max_pool')
            return conv_


def fc(X, unit_size, activation='sigmoid', device='/GPU:0', name='full_connect', keep_prob=0):
    """
    :param X: input tensor
    :param unit_size: the unit size of this layer.
    :param activation: activation function. you can use: 'relu', 'sigmoid', 'softmax', 'relu6', 'tanh'
    :param name: layer's name.
    :return:
    """
    with tf.name_scope(name):
        with tf.device(device):
            x_shape = X.shape.as_list()
            if len(X.shape.as_list()) != 2:
                raise Exception("The input tensor 'X' must have 2 dimension")
            w = tf.random.normal([X.shape.as_list()[1], unit_size], name='weight')
            b = tf.random.normal([unit_size], name='bias')
            fc_ = tf.matmul(X, w, name='matmul')
            fc_ = tf.nn.bias_add(fc_, b)
            if activation == 'sigmoid':
                fc_ = tf.nn.sigmoid(fc_, activation)
            elif activation == 'relu':
                fc_ = tf.nn.relu(fc_, activation)
            elif activation == 'relu6':
                fc_ = tf.nn.relu6(fc_, activation)
            elif activation == 'tanh':
                fc_ = tf.nn.tanh(fc_, activation)
            elif activation == 'softmax':
                fc_ = tf.nn.softmax(fc_, name=activation)
            fc_ = tf.nn.dropout(fc_, keep_prob, name='dropout')
            return fc_


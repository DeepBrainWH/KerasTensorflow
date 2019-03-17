import tensorflow as tf
import numpy as np

config = tf.ConfigProto(allow_soft_placement=True)
device0 = '/GPU:0'
device1 = '/GPU:0'
device2 = '/GPU:0'
device3 = '/GPU:0'
# 卷积操作：---
# 设输入图像尺寸为W, 卷积核尺寸为：F, 步长：S
# 如果 Padding='SAME'，输出尺寸为： ceil(W / S)
# 如果 Padding='VALID'，输出尺寸为：ceil((W - F + 1) / S)
# ceil操作为向上取整


with tf.name_scope("input"):
    with tf.device(device0):
        X_ = tf.placeholder(tf.float32, shape=[None, 306, 408, 3], name="X")
        y_ = tf.placeholder(tf.float32, shape=[None, 9], name="y")

# ========================conv 1============================
with tf.name_scope("conv_1"):
    with tf.device(device1):
        kernel = tf.random.normal([5, 5, 3, 32], name="kernel_1")
        bias = tf.random.normal([32])
        conv_1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X_, kernel, [1, 2, 2, 1], 'SAME'), bias), name='conv_1_1')

        pool_1_1 = tf.nn.max_pool(X_, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name="max_pool_1_1")

        kernel2 = tf.random.normal([3, 3, 3, 16], name='kernel_2')
        bias2 = tf.random.normal([16], name='bias2')
        conv_1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X_, kernel2, [1, 1, 1, 1], 'SAME'), bias2), name="conv_1_2")
        pool_1_2 = tf.nn.max_pool(conv_1_2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='max_pool_1_2')

        concat_value = tf.concat([conv_1_1, pool_1_1, pool_1_2], 3, 'concat_tensor')
        # size: (None, 153, 204, 51)
        conv_1_shape = concat_value.shape.as_list()

# ========================conv 2==============================
with tf.name_scope("conv_2"):
    with tf.device(device1):
        kernel = tf.random.normal([5, 5, conv_1_shape[3], 64], name="kernel_1")
        bias = tf.random.normal([64])
        conv_2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X_, kernel, [1, 2, 2, 1], 'SAME'), bias), name='conv_1_1')

        pool_2_1 = tf.nn.max_pool(X_, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name="max_pool_1_1")

        kernel2 = tf.random.normal([3, 3, conv_1_shape[3], 16], name='kernel_2')
        bias2 = tf.random.normal([16], name='bias2')
        conv_2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X_, kernel2, [1, 1, 1, 1], 'SAME'), bias2), name="conv_1_2")
        pool_2_2 = tf.nn.max_pool(conv_2_2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='max_pool_1_2')

        concat_value = tf.concat([conv_2_1, pool_2_1, pool_2_2], 3, 'concat_tensor')
        conv_2_shape = concat_value.shape.as_list()

with tf.name_scope("conv3"):
    with tf.device(device2):
        kernel = tf.random.norma()

with tf.name_scope("full_connection_1"):
    with tf.device("/GPU:0"):
        w1 = tf.Variable(tf.random.normal([306, 128]), name="w1")
        b1 = tf.Variable(tf.random.normal([128]), name="b1")
        result = tf.nn.bias_add(tf.matmul(X_, w1, name="mat_mul"), b1, name="biase_add")

with tf.name_scope("loss_value_calculate"):
    with tf.device("/GPU:0"):
        loss = tf.losses.mean_squared_error(result, y_)
        tf.summary.scalar("loss", loss)
        train_ = tf.train.AdamOptimizer().minimize(loss, name="optimize")

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

x__ = np.random.randn(100, 306)
y__ = np.random.randn(100, 128)

print(x__.shape)
print(y__.shape)
with tf.Session(config=config) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("./log", sess.graph)
    for i in range(1000):
        sess.run(train_, feed_dict={X_: x__, y_: y__})
        if i % 10 == 0:
            summary = sess.run(merged, feed_dict={X_: x__, y_: y__})
            train_writer.add_summary(summary, i)

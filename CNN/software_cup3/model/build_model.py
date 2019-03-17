from __future__ import absolute_import
from get_data import GET_DATA
import get_data
import tensorflow as tf
import numpy as np


class Model:
    def __init__(self):
        self.X = None
        self.y = None

    def build_model(self):
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        X_ = tf.placeholder(dtype=tf.float32, shape=[None, 306, 408, 3])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 8])
        keep_prob = tf.placeholder(dtype=tf.float32)

        w1 = tf.Variable(tf.random.normal(shape=[5, 5, 3, 64], mean=0.0, stddev=0.1))
        b1 = tf.Variable(tf.random.normal(shape=[64], mean=0.0, stddev=0.1))

        w2 = tf.Variable(tf.random.normal(shape=[5, 5, 64, 64], mean=0.0, stddev=0.1))
        b2 = tf.Variable(tf.random.normal(shape=[64], mean=0.0, stddev=0.1))

        w3 = tf.Variable(tf.random.normal(shape=[5, 5, 64, 64], mean=0.0, stddev=0.1))
        b3 = tf.Variable(tf.random.normal(shape=[64], mean=0.0, stddev=0.1))

        w4 = tf.Variable(tf.random.normal(shape=[5, 5, 64, 16], mean=0.0, stddev=0.1))
        b4 = tf.Variable(tf.random.normal(shape=[16], mean=0.0, stddev=0.1))

        with tf.variable_scope("conv1"):
            conv1 = tf.nn.conv2d(X_, w1, [1, 1, 1, 1], "SAME")
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
            res1 = tf.nn.max_pool(relu1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        with tf.variable_scope("conv2"):
            conv2 = tf.nn.conv2d(res1, w2, [1, 1, 1, 1], "SAME")
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2, name="biase_add"), name="activation")
            res2 = tf.nn.max_pool(relu2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name="max_poll")

        with tf.variable_scope("conv3"):
            conv3 = tf.nn.conv2d(res2, w3, [1, 1, 1, 1], "SAME")
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3, name="biase_add"), name="activation")
            res3 = tf.nn.max_pool(relu3, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name="max_poll")

        with tf.variable_scope("conv4"):
            conv4 = tf.nn.conv2d(res3, w4, [1, 1, 1, 1], "SAME")
            relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4, name="biase_add"), name="activation")
            res4 = tf.nn.max_pool(relu4, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name="max_poll")

        res4 = tf.reshape(res4, (get_data.BATCH_SIZE, -1))
        wf_1 = tf.Variable(tf.random.normal(shape=[8320, 128], mean=0.0, stddev=0.1), name="wf1")
        bf_1 = tf.Variable(tf.random.normal(shape=[128], mean=0.0, stddev=0.1), name="bf1")

        wf_2 = tf.Variable(tf.random.normal(shape=[128, 512], mean=0.0, stddev=0.1), name="wf2")
        bf_2 = tf.Variable(tf.random.normal(shape=[512], mean=0.0, stddev=0.1), name="bf2")

        wf_3 = tf.Variable(tf.random.normal(shape=[512, 8], mean=0.0, stddev=0.1), name="wf3")
        bf_3 = tf.Variable(tf.random.normal(shape=[8], mean=0.0, stddev=0.1), name="bf3")

        with tf.variable_scope("fc1"):
            fc1 = tf.nn.bias_add(tf.matmul(res4, wf_1), bf_1, name="biase_add_fc1")
            fc1 = tf.nn.relu(fc1, "activation")
            dropout1 = tf.nn.dropout(fc1, keep_prob)

        with tf.variable_scope("fc2"):
            fc2 = tf.nn.bias_add(tf.matmul(dropout1, wf_2), bf_2, name="biase_add_fc2")
            fc2 = tf.nn.relu(fc2, "activation")
            dropout2 = tf.nn.dropout(fc2, keep_prob)

        with tf.variable_scope("fc3"):
            fc3 = tf.nn.bias_add(tf.matmul(dropout2, wf_3), bf_3, name="biase_add_fc3")
            fc3 = tf.nn.relu(fc3, name="relu_activation")

        loss_function = self.my_loss_function(y, fc3)
        tf.summary.scalar("loss", loss_function)
        with tf.device("/device:GPU:0"):
            train_op = tf.train.AdamOptimizer(0.001).minimize(loss_function)

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.008)
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./logs", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(1000):
            tmp_loss = 0
            for _x, _y in GET_DATA.get_batches_data():
                train_x = (_x - np.min(_x)) / ((np.max(_x) - np.min(_x)))
                train_y = _y
                sess.run(train_op, feed_dict={X_: train_x, y: train_y, keep_prob: 0.2})
                if i % 10 ==0:
                    summary = sess.run(merged, feed_dict={X_: train_x, y: train_y, keep_prob: 0.2})
                    train_writer.add_summary(summary, i)
                    tmp_loss += sess.run(loss_function, feed_dict={X_: train_x, y: train_y, keep_prob: 0.2})
            saver.save(sess, "./checkpoint/savemodel")
            if i % 10 == 0:
                print("step %d , the loss value is: %.2f" % (i, tmp_loss / 3))

    def my_loss_function(self, y, y_):
        """
        :param y: real value
        :param y_: predict value
        :return: loss value.
        """
        if y[0] == 0:
            loss_value = tf.reduce_mean(tf.square(y_[0] - y[0]))
        else:
            loss_value = tf.reduce_mean(tf.square(y_-y))
        return loss_value


if __name__ == '__main__':
    model = Model()
    model.build_model()

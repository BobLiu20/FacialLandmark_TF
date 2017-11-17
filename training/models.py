import sys
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, fully_connected as fc
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework import arg_scope

slim = tf.contrib.slim

def fanet8ss_conv_1_1_16_16_16_exp(input_tensor, point_label):
    normal_tensor = input_tensor/127.5 - 1.0
    h_conv1 = conv2d(normal_tensor, 16, 3, stride=2)
    h_conv2 = conv2d(h_conv1, 16, 1)
    h_conv3 = conv2d(h_conv2, 16, 3, stride=2)
    h_conv4 = conv2d(h_conv3, 16, 1)
    h_conv5 = conv2d(h_conv4, 32, 3, stride=2)
    h_conv6 = conv2d(h_conv5, 64, 3)
    h_conv7 = conv2d(h_conv6, 64, 3, stride=2)
    h_conv8 = conv2d(h_conv7, 128, 3)
    h_pool1 = max_pool2d(h_conv8, 2, 2)
    # h_pool1.set_shape([None, 3, 3, 128])
    # h_pool1_flat = tf.reshape(h_pool1, [-1, 3*3*128])
    h_pool1_flat = flatten(h_pool1)
    h_fc1 = fc(h_pool1_flat, 512)
    point = fc(h_fc1, point_label, activation_fn=None)
    return point

def get_l2_loss(logits, labels, batch_size):
    loss = tf.nn.l2_loss(tf.subtract(logits, labels))
    loss = tf.truediv(loss, tf.constant(batch_size, dtype=tf.float32))
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(loss)
    return total_loss

def init(model, images, num_labels):
    func = getattr(sys.modules["models"], model)
    return func(images, num_labels)


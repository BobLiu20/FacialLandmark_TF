import sys
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, fully_connected as fc
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework import arg_scope

slim = tf.contrib.slim

# models
def fanet8ss_inference(inputs, num_label, **kwargs): 
    with tf.variable_scope('fanet8ss_inference') as scope:
        inputs = inputs/127.5-1.0
        h_conv1 = conv2d(inputs, 16, 3, stride=2)

        h_conv2 = conv2d(h_conv1, 32, 3)
        h_conv3 = conv2d(h_conv2, 32, 3, stride=2)

        h_conv4 = conv2d(h_conv3, 32, 3)
        h_conv5 = conv2d(h_conv4, 32, 3, stride=2)

        h_conv6 = conv2d(h_conv5, 64, 3)
        h_conv7 = conv2d(h_conv6, 64, 3, stride=2)

        h_conv8 = conv2d(h_conv7, 128, 3)
        h_pool1 = max_pool2d(h_conv8, 2, 2)
        h_pool1_flat = flatten(h_pool1)
        h_fc1 = fc(h_pool1_flat, 512)
        h_fc2 = fc(h_fc1, num_label, activation_fn=tf.nn.sigmoid)
        return h_fc2

def fanet8ss_conv_1_1_16_16_16_exp(input_tensor, num_label, **kwargs):
    with tf.variable_scope('fanet8ss_conv_1_1_16_16_16_exp') as scope:
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
        h_pool1_flat = flatten(h_pool1)
        h_fc1 = fc(h_pool1_flat, 512)
        point = fc(h_fc1, num_label, activation_fn=None)
        return point

# loss
def get_l2_loss(logits, labels):
    loss = tf.nn.l2_loss(tf.subtract(logits, labels))
    loss = tf.truediv(loss, tf.constant(batch_size, dtype=tf.float32))
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(loss)
    tf.add_to_collection(tf.GraphKeys.LOSSES, total_loss)
    return total_loss

def get_mse_loss(logits, labels):
    return tf.losses.mean_squared_error(labels, logits)

# init
def init(model, images, num_labels, **kwargs):
    func = getattr(sys.modules["models"], model)
    return func(images, num_labels, **kwargs)


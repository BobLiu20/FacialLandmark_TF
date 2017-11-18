import sys
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, fully_connected as fc
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework import arg_scope

slim = tf.contrib.slim

def prelu(tensor):
    alphas = tf.get_variable('alpha', tensor.get_shape()[-1], 
                            initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(tensor)
    neg = -alphas * tf.nn.relu(-tensor)
    return pos+neg

def fanet8ss_conv_1_1_16_16_16_exp(input_tensor, num_label, **kwargs):
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

def fanet8ss_conv_1_1_16_16_16_exp_3fc(inputs, num_label, reuse=True, **kwargs): 
    with slim.arg_scope([slim.conv2d], activation_fn=prelu, 
                        weights_initializer=initializers.xavier_initializer(), 
                        padding='SAME', reuse=reuse):
        net = inputs/127.5-1.0
        net = slim.conv2d(net, 16, [3,3], stride=2, scope='conv_1')
        net = slim.conv2d(net, 16, [1,1], scope='conv_2')
        net = slim.conv2d(net, 16, [3,3], stride=2, scope='conv_3')
        net = slim.conv2d(net, 16, [1,1], scope='conv_4')
        net = slim.conv2d(net, 32, [3,3], stride=2, scope='conv_5')
        net = slim.conv2d(net, 64, [3,3], scope='conv_6')
        net = slim.conv2d(net, 64, [3,3], stride=2, scope='conv_7')
        net = slim.conv2d(net, 128, [3,3], scope='conv_8')
        net = slim.max_pool2d(net, [2,2], padding='VALID', scope='max_pooling_1')
        net = slim.flatten(net, scope='flat_1')
    with slim.arg_scope([slim.fully_connected], activation_fn=prelu, 
                        weights_initializer=initializers.xavier_initializer(), reuse=reuse):
        net = slim.fully_connected(net, 256, scope='fc/fc_0')
        logits_pts = slim.fully_connected(net, 128, scope='fc/fc_3')
        logits_pts = slim.fully_connected(logits_pts, num_label, activation_fn=None, scope='fc/fc_4')
    return logits_pts

def fanet8ss_inference(inputs, num_label, **kwargs): 
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

def inception_v3(inputs, num_label, **kwargs):
    from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3, inception_v3_arg_scope
    arg_scope = inception_v3_arg_scope()
    with tf.contrib.slim.arg_scope(arg_scope):
        logits, _ = inception_v3(inputs, num_label, is_training=kwargs['is_training'])
    return logits

def get_l2_loss(logits, labels, batch_size):
    loss = tf.nn.l2_loss(tf.subtract(logits, labels))
    loss = tf.truediv(loss, tf.constant(batch_size, dtype=tf.float32))
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(loss)
    return total_loss

def init(model, images, num_labels, **kwargs):
    func = getattr(sys.modules["models"], model)
    return func(images, num_labels, **kwargs)


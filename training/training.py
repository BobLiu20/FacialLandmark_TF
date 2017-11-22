# coding='utf-8'
import os
import sys
import argparse
import numpy as np
import time
import datetime

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'common'))
from batch_reader import BatchReader
import models

def tower_loss(scope, images, labels, model_name, num_labels):
    # Build inference Graph.
    logits = models.init(model_name, images, num_labels, is_training=True)
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = models.get_mse_loss(logits, labels)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(prefix, **arg_dict):
    batch_size = arg_dict['batch_size']
    num_labels = arg_dict['landmark_type'] * 2
    img_size = arg_dict['img_size']
    gpu_list = map(int, arg_dict['gpu_device'].split(','))
    assert (batch_size % len(gpu_list) == 0), "Batch size must exact division by gpu nums"

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # data input
        images = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, 3])
        labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
        images_split = tf.split(images, len(gpu_list), axis=0)
        labels_split = tf.split(labels, len(gpu_list), axis=0)
        # Create a variable to count the number of train() calls.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(arg_dict['learning_rate'],
                                        global_step,
                                        30000,
                                        0.8,
                                        staircase=True)
        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.AdamOptimizer(lr)
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(len(gpu_list)):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("landmarks", i)) as scope:
                        loss = tower_loss(scope, images_split[i], labels_split[i], arg_dict['model'], num_labels)
                        tf.get_variable_scope().reuse_variables()
                        # Calculate the gradients for the batch of data on this tower.
                        grads = optimizer.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                          gpu_options = tf.GPUOptions(allow_growth=True)))
        sess.run(init)
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        if arg_dict['restore_ckpt']:
            variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, arg_dict['restore_ckpt'])
            print ('Resume-trained model restored from: %s' % arg_dict['restore_ckpt'])

        print ("Start to training...")
        # batch generator
        _batch_reader = BatchReader(**arg_dict)
        _batch_generator = _batch_reader.batch_generator()
        start_time = time.time()
        while not _batch_reader.should_stop():
            batch = _batch_generator.next()
            _, _loss, _step, _lr = sess.run([train_op, loss, global_step, lr], 
                                         feed_dict={images: batch[0], labels: batch[1]})
            if _step % 10 == 0:
                end_time = time.time()
                cost_time, start_time = end_time - start_time, end_time
                sample_per_sec = int(10 * batch_size / cost_time)
                sec_per_step = cost_time / 10.0
                print ('[%s] epochs: %d, step: %d, lr: %f, landmark_loss: %.6f, sample/s: %d, sec/step: %.3f' % (
                       datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 
                       _batch_reader.get_epoch(), _step, _lr, _loss, sample_per_sec, sec_per_step))
            if _step % 1024 == 0:
                checkpoint_path = os.path.join(prefix, 'model.ckpt')
                saver.save(sess, checkpoint_path)
                print ('Saved checkpoint to %s' % checkpoint_path)
        checkpoint_path = os.path.join(prefix, 'model.ckpt')
        saver.save(sess, checkpoint_path)
        print ('\nReview training parameter:\n%s\n'%(str(arg_dict)))
        print ('Saved checkpoint to %s' % checkpoint_path)
        print ('Bye Bye!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_paths', type=str, nargs='+',
                        default='/world/data-c9/liubofang/dataset_original/CelebA/full_path_zf_bbox_pts.txt')
    parser.add_argument('--working_root', type=str, default='/world/data-c9/liubofang/training/landmarks/celeba')
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training. Total for all gpus.")
    parser.add_argument('--landmark_type', type=int, default=5, help="The number of points. 5 or 83")
    parser.add_argument('--max_epoch', type=int, default=1000, help="Training will be stoped in this case.")
    parser.add_argument('--img_size', type=int, default=128, help="The size of input for model")
    parser.add_argument('--max_angle', type=int, default=10, help="Use for image augmentation")
    parser.add_argument('--process_num', type=int, default=20, help="The number of process to preprocess image.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="lr")
    parser.add_argument('--model', type=str, default='fanet8ss_inference', help="Model name. Check models.py")
    parser.add_argument('--restore_ckpt', type=str, help="Resume training from special ckpt.")
    parser.add_argument('--try', type=int, default=0, help="Saving path index")
    parser.add_argument('--gpu_device', type=str, default='4,5,6,7', help="GPU index. Support Multi GPU. eg: 1,2,3")
    parser.add_argument('--img_format', type=str, default='RGB', help="The color format for training.")
    parser.add_argument('--buffer2memory', type=bool, default=False, 
                        help="Read all image to memory to speed up training. Make sure enough memory in your device.")
    arg_dict = vars(parser.parse_args())
    prefix = '%s/%d/%s/size%d_angle%d_try%d' % (
        arg_dict['working_root'], arg_dict['landmark_type'], arg_dict['model'], 
        arg_dict['img_size'], arg_dict['max_angle'], arg_dict['try'])
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # set up TF environment
    os.environ['CUDA_VISIBLE_DEVICES']=arg_dict['gpu_device']
    global tf
    import tensorflow as tf

    train(prefix, **arg_dict)

if __name__ == "__main__":
    main()


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

def train(prefix, **arg_dict):
    batch_size = arg_dict['batch_size']
    num_labels = arg_dict['landmark_type'] * 2
    img_size = arg_dict['img_size']
    # batch generator
    _batch_reader = BatchReader(**arg_dict)
    _batch_generator = _batch_reader.batch_generator()

    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, 3])
        point_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])

        logits = models.init(arg_dict['model'], images, num_labels, is_training=True)

        loss = models.get_l2_loss(logits, point_labels, batch_size)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(arg_dict['learning_rate'], global_step, 30000, 0.5, staircase=True)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(allow_growth=True)
            )) 
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())

        if arg_dict['restore_ckpt']:
            variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, arg_dict['restore_ckpt'])
            print ('Resume-trained model restored from: %s' % arg_dict['restore_ckpt'])

        tf.train.write_graph(sess.graph.as_graph_def(), '.', os.path.join(prefix,'graph_struct.txt'))

        print ("Start to training...")
        start_time = time.time()
        while not _batch_reader.should_stop():
            with tf.device('/gpu:0'):
                batch = _batch_generator.next()
                _, ploss, step, lr = sess.run([train_op, loss, global_step, learning_rate], 
                                             feed_dict={images: batch[0], point_labels: batch[1]})
                if step % 10 == 0:
                    end_time = time.time()
                    cost_time, start_time = end_time - start_time, end_time
                    sample_per_sec = int(10 * batch_size / cost_time)
                    sec_per_step = cost_time / 10.0
                    print ('[%s] epochs: %d, step: %d, lr: %f, landmark_loss: %.4f, sample/s: %d, sec/step: %.3f' % (
                           datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 
                           _batch_reader.get_epoch(), step, lr, ploss, sample_per_sec, sec_per_step))
            if step % 1024 == 0:
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
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training")
    parser.add_argument('--landmark_type', type=int, default=5, help="The number of points. 5 or 83")
    parser.add_argument('--max_epoch', type=int, default=1000, help="Training will be stoped in this case.")
    parser.add_argument('--img_size', type=int, default=128, help="The size of input for model")
    parser.add_argument('--max_angle', type=int, default=10, help="Use for image augmentation")
    parser.add_argument('--process_num', type=int, default=20, help="The number of process to preprocess image.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="lr")
    parser.add_argument('--model', type=str, default='fanet8ss_inference', help="Model name. Check models.py")
    parser.add_argument('--restore_ckpt', type=str, help="Resume training from special ckpt.")
    parser.add_argument('--try', type=int, default=0, help="Saving path index")
    parser.add_argument('--gpu_device', type=str, default='7', help="GPU index")
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


#coding=utf8
import os
import sys
import cv2
import numpy as np

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'training'))
import models

class Predictor():
    def __init__(self, _arg_dict):
        self._arg_dict = _arg_dict
        # set up TF environment
        os.environ['CUDA_VISIBLE_DEVICES'] = self._arg_dict['gpu_device']
        global tf
        import tensorflow as tf
        self.__init_model()

    def __init_model(self):
        with tf.Graph().as_default():
            self.placeholder = tf.placeholder(dtype=tf.float32, 
                shape=[None, self._arg_dict['img_size'], self._arg_dict['img_size'], 3])
            self.logits = models.init(self._arg_dict['model'], self.placeholder, self._arg_dict['landmark_type']*2)
            saver = tf.train.Saver()
            with tf.device('/gpu:0'):
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
                ckpt_path = self._arg_dict['model_path']
                if tf.train.checkpoint_exists(ckpt_path):
                    saver.restore(self.sess, ckpt_path)
                else:
                    raise Exception("model_path inexistence")

    def predict(self, img):
        points = self.sess.run([self.logits], feed_dict={self.placeholder: img})
        return np.array(points).reshape(-1, 2)


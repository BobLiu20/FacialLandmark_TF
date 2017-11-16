#coding=utf-8
import os
import sys
import numpy as np
import cv2
import math
import signal
import random
from multiprocessing import Process, Queue

from landmark_augment import LandmarkAugment
from landmark_helper import LandmarkHelper

process_list = []
def handler(sig_num, stack_frame):
    for p in process_list:
        p.terminate()
    sys.exit()
signal.signal(signal.SIGINT, handler)

class BatchReader():
    def __init__(self, **kwargs):
        # param
        self._kwargs = kwargs
        self._batch_size = kwargs['batch_size']
        self._process_num = kwargs['process_num']
        # total lsit
        self._sample_list = [] # each item: (filepath, landmarks, ...)
        self._total_sample = 0
        # real time buffer
        self._output_queue = []
        for i in range(self._process_num):
            self._output_queue.append(Queue(maxsize=3)) # for each process
        # epoch
        self._idx_in_epoch = 0
        self._curr_epoch = 0
        # start buffering
        self._start_buffering(kwargs['input_paths'], kwargs['landmark_type'])

    def batch_generator(self):
        __curr_queue = 0
        while True:
            self.__update_epoch()
            while True:
                __curr_queue += 1
                if __curr_queue >= self._process_num:
                    __curr_queue = 0
                try:
                    image_list, landmarks_list = self._output_queue[__curr_queue].get(block=True, timeout=0.01)
                    break
                except Exception as ex:
                    pass
            yield image_list, landmarks_list

    def get_epoch(self):
        return self._curr_epoch

    def _start_buffering(self, input_paths, landmark_type):
        if type(input_paths) in [str, unicode]:
            input_paths = [input_paths]
        for input_path in input_paths:
            for line in open(input_path):
                info = LandmarkHelper.parse(line, landmark_type)
                self._sample_list.append(info)
        self._total_sample = len(self._sample_list)
        num_per_process = int(math.ceil(self._total_sample / float(self._process_num)))
        for idx, offset in enumerate(range(0, self._total_sample, num_per_process)):
            p = Process(target=self._process, args=(idx, self._sample_list[offset: offset+num_per_process]))
            p.start()
            process_list.append(p)

    def _process(self, idx, sample_list):
        __landmark_augment = LandmarkAugment()
        # read all image to memory to speed up!
        if self._kwargs['buffer2memory']:
            print ("Start to read image to memory! Count=%d"%(len(sample_list)))
            sample_list = __landmark_augment.mini_crop_by_landmarks(sample_list, 4.5, self._kwargs['img_format'])
            print ("Read all image to memory finish!")
        sample_cnt = 0 # count for one batch
        image_list, landmarks_list = [], [] # one batch list
        while True:
            for sample in sample_list:
                # preprocess
                if type(sample[0]) in [str, unicode]:
                    image = cv2.imread(sample[0])
                    if self._kwargs['img_format'] == 'RGB':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.imdecode(sample[0], cv2.CV_LOAD_IMAGE_COLOR)
                landmarks = sample[1].copy()# keep deep copy
                scale_range = (2.7, 3.3)
                image_new, landmarks_new = __landmark_augment.augment(image, landmarks, self._kwargs['img_size'],
                                            self._kwargs['max_angle'], scale_range)
                # sent a batch
                sample_cnt += 1
                image_list.append(image_new)
                landmarks_list.append(landmarks_new)
                if sample_cnt >= self._kwargs['batch_size']:
                    self._output_queue[idx].put((np.array(image_list), np.array(landmarks_list)))
                    sample_cnt = 0
                    image_list, landmarks_list = [], []
            np.random.shuffle(sample_list)

    def __update_epoch(self):
        self._idx_in_epoch += self._batch_size
        if self._idx_in_epoch > self._total_sample:
            self._curr_epoch += 1
            self._idx_in_epoch = 0

# use for unit test
if __name__ == '__main__':
    kwargs = {
        'input_paths': "/world/data-c9/liubofang/dataset_original/CelebA/full_path_zf_bbox_pts.txt",
        'landmark_type': 5,
        #'input_paths': "/world/data-c22/AR/landmarks/CelebA/CelebA_19w_washed.txt",
        #'landmark_type': 83,
        'batch_size': 512,
        'process_num': 30,
        'img_format': 'RGB',
        'img_size': 112,
        'max_angle': 10,
        'buffer2memory': True,
    }
    b = BatchReader(**kwargs)
    g = b.batch_generator()
    output_folder = "output_tmp/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    import time
    start_time = time.time()
    for i in range(1000000):
        end_time = time.time()
        print ("get new batch...step: %d. epoch: %d. cost: %.3f"%(
                i, b.get_epoch(), end_time-start_time))
        start_time = end_time
        batch_image, batch_landmarks = g.next()
        for idx, (image, landmarks) in enumerate(zip(batch_image, batch_landmarks)):
            if idx > 20: # only see first 10
                break
            landmarks = landmarks.reshape([-1, 2])
            for l in landmarks:
                ii = tuple(l * (kwargs['img_size'], kwargs['img_size']))
                cv2.circle(image, (int(ii[0]), int(ii[1])), 2, (0,255,0), -1)
            cv2.imwrite("%s/%d.jpg"%(output_folder, idx), image)
    print ("Done...Press ctrl+c to exit me")


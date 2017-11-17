#coding=utf8
import os
import sys
import argparse
import cv2
import numpy as np

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
from predictor import Predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
        default='/world/data-c9/liubofang/training/landmarks/celeba/working/5/fanet8ss_conv_1_1_16_16_16_exp/fold_0/model.ckpt')
    parser.add_argument('--model', type=str, default='fanet8ss_conv_1_1_16_16_16_exp')
    parser.add_argument('--landmark_type', type=int, default=5) #5 or 83
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--images_folder', type=str, 
        default='/home/liubofang/other_script/bob_training_set_自己的训练集制作/images_crop')
    parser.add_argument('--img_format',type=str,default='RGB')
    parser.add_argument('--gpu_device', type=str, default='7')
    arg_dict = vars(parser.parse_args())
    # output folder
    out_folder = os.path.join(MY_DIRNAME, "output_tmp")
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    # predict
    _testing = Predictor(arg_dict)
    for idx, im_path in enumerate(os.listdir(arg_dict['images_folder'])):
        im_path = os.path.join(arg_dict['images_folder'], im_path)
        img = cv2.imread(im_path)
        if arg_dict['img_format'] == 'RGB':
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (arg_dict['img_size'], arg_dict['img_size']))
        h, w = img.shape[:2]
        new_img = img.copy()
        new_img = np.array([new_img])
        points = _testing.predict(new_img)
        for (x, y) in points:
            y = int(y * h) 
            x = int(x * w) 
            cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
        cv2.imwrite(os.path.join(out_folder, os.path.basename(im_path)), img)
        print ("predicting %s"%im_path)


"""
# Script for evaluating F score and mIOU 
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Usage: for academic and educational purposes only
"""
from __future__ import print_function, division
import ntpath
import cv2
import numpy as np
from skimage import data
import skimage.transform as trans
# local libs
from utils.data_utils import getPaths
from utils.measure_utils import db_eval_boundary, IoU_bin

## experiment directories
obj_cat = "HD/" # sub-dir  ["RI/", "FV/", "WR/", "RO/", "HD/"]
test_dir = "sample_test/masks/"
real_mask_dir = test_dir + obj_cat # real labels
gen_mask_dir = "sample_test/output/" + obj_cat # generated labels

## input/output shapes
im_w, im_h, chan = 320, 240, 3 
im_shape = (im_h, im_w, 3)
mask_shape = (im_h, im_w)

# for reading and scaling input images
def read_and_bin(im_path):
    img = data.imread(im_path, as_grey=True)
    img = trans.resize(img, mask_shape)
    #img[img > 0.5] = 1
    #img[img <= 0.5] = 0
    return img

# accumulate F1/iou values in the lists
Ps, Rs, F1s, IoUs = [], [], [], []
for p in getPaths(real_mask_dir):
    img_name = ntpath.basename(p)
    img_name = img_name.split('.')[0]
    gen_path = gen_mask_dir + img_name + ".bmp"
    real, gen = read_and_bin(p), read_and_bin(gen_path)
    if (np.sum(real)>0):
        precision, recall, F1 = db_eval_boundary(real, gen)
        iou = IoU_bin(real, gen)
        print ("{0}:>> P: {1}, R: {2}, F1: {3}, IoU: {4}".format(img_name, precision, recall, F1, iou))
        Ps.append(precision) 
        Rs.append(recall)
        F1s.append(F1)
        IoUs.append(iou)

# print F-score and mIOU in [0, 100] scale
print ("Avg. F: {0}".format(100.0*np.mean(F1s)))
print ("Avg. IoU: {0}".format(100.0*np.mean(IoUs)))
    


"""
# Script for evaluating F score and mIOU 
"""
from __future__ import print_function, division
import ntpath
import numpy as np
from PIL import Image
# local libs
from utils.data_utils import getPaths
from utils.measure_utils import db_eval_boundary, IoU_bin

## experiment directories
obj_cat = "RO/" # sub-dir  ["RI/", "FV/", "WR/", "RO/", "HD/"]
test_dir = "sample_test/masks/"
test_dir = "/home/xahid/datasets/released/SUIM/TEST/masks/"
real_mask_dir = test_dir + obj_cat # real labels
gen_mask_dir = "sample_test/output/" + obj_cat # generated labels

## input/output shapes
im_res = (320, 240)

# for reading and scaling input images
def read_and_bin(im_path):
    img = Image.open(im_path).resize(im_res)
    img = np.array(img)/255.
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

# accumulate F1/iou values in the lists
Ps, Rs, F1s, IoUs = [], [], [], []
gen_paths = sorted(getPaths(gen_mask_dir))
real_paths = sorted(getPaths(real_mask_dir))
for gen_p, real_p in zip(gen_paths, real_paths):
    gen, real = read_and_bin(gen_p), read_and_bin(real_p)
    if (np.sum(real)>0):
        precision, recall, F1 = db_eval_boundary(real, gen)
        iou = IoU_bin(real, gen)
        #print ("{0}:>> P: {1}, R: {2}, F1: {3}, IoU: {4}".format(gen_p, precision, recall, F1, iou))
        Ps.append(precision) 
        Rs.append(recall)
        F1s.append(F1)
        IoUs.append(iou)

# print F-score and mIOU in [0, 100] scale
print ("Avg. F: {0}".format(100.0*np.mean(F1s)))
print ("Avg. IoU: {0}".format(100.0*np.mean(IoUs)))
    


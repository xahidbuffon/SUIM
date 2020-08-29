"""
# Test script for the SUIM-Net
    # for 5 object categories: HD, FV, RO, RI, WR 
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Usage: for academic and educational purposes only
"""
from __future__ import print_function, division
import os
import time
import ntpath
import numpy as np
from scipy import misc
from skimage import data
import skimage.transform as trans
# local libs
from model import SUIM_Net
from utils.data_utils import getPaths

## experiment directories
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/images/"
test_dir = "sample_test/images/"

## sample and ckpt dir
samples_dir = "sample_test/output/"
RO_dir = samples_dir + "RO/"
FB_dir = samples_dir + "FV/"
WR_dir = samples_dir + "WR/"
HD_dir = samples_dir + "HD/"
RI_dir = samples_dir + "RI/" 
if not os.path.exists(samples_dir): os.makedirs(samples_dir)
if not os.path.exists(RO_dir): os.makedirs(RO_dir)
if not os.path.exists(FB_dir): os.makedirs(FB_dir)
if not os.path.exists(WR_dir): os.makedirs(WR_dir)
if not os.path.exists(HD_dir): os.makedirs(HD_dir)
if not os.path.exists(RI_dir): os.makedirs(RI_dir)

## input/output shapes
im_w, im_h, chan = 320, 240, 3
im_shape = (im_h, im_w, 3)

model = SUIM_Net(im_res=(im_h, im_w), n_classes=5).model
print (model.summary())
model_ckpt_name = "sample_test/ckpt_seg_5obj.hdf5"
model.load_weights(model_ckpt_name)


def testGenerator():
    # test all images in the directory
    assert os.path.exists(test_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(test_dir):
        # read and scale inputs
        img = data.imread(p, as_grey=False)
        img = trans.resize(img, im_shape)
        img = np.expand_dims(img, axis=0)
        # inference
        out_img = model.predict(img)
        # thresholding
        out_img[out_img>0.5] = 1.
        out_img[out_img<=0.5] = 0.
        print ("tested: {0}".format(p))
        # get filename
        img_name = ntpath.basename(p)
        img_name = img_name.split('.')[0]
        # save individual output masks
        ROs = np.reshape(out_img[0,:,:,0], (im_h, im_w))
        FVs = np.reshape(out_img[0,:,:,1], (im_h, im_w))
        HDs = np.reshape(out_img[0,:,:,2], (im_h, im_w))
        RIs = np.reshape(out_img[0,:,:,3], (im_h, im_w))
        WRs = np.reshape(out_img[0,:,:,4], (im_h, im_w))
        misc.imsave(RO_dir+img_name+'.bmp', ROs)
        misc.imsave(FB_dir+img_name+'.bmp', FVs)
        misc.imsave(HD_dir+img_name+'.bmp', HDs)
        misc.imsave(RI_dir+img_name+'.bmp', RIs)
        misc.imsave(WR_dir+img_name+'.bmp', WRs)

# test images
testGenerator()







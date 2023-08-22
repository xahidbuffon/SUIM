"""
# Test script for the SegNet (ResNet50)
    # for 5 object categories: HD, FV, RO, RI, WR 
# See https://arxiv.org/pdf/2004.01241.pdf  
"""
from __future__ import print_function, division
import os
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
# local libs
from models.segnet import resnet50_segnet
from utils.data_utils import getPaths

## experiment directories
#test_dir = "/mnt/data1/ImageSeg/suim/TEST/images/"
test_dir = "data/test/images/"

## sample and ckpt dir
samples_dir = "data/test/outputSeg/"
RO_dir = samples_dir + "RO/"
FB_dir = samples_dir + "FV/"
WR_dir = samples_dir + "WR/"
HD_dir = samples_dir + "HD/"
RI_dir = samples_dir + "RI/" 
if not exists(samples_dir): os.makedirs(samples_dir)
if not exists(RO_dir): os.makedirs(RO_dir)
if not exists(FB_dir): os.makedirs(FB_dir)
if not exists(WR_dir): os.makedirs(WR_dir)
if not exists(HD_dir): os.makedirs(HD_dir)
if not exists(RI_dir): os.makedirs(RI_dir)

## input/output shapes
im_res_ = (320, 256, 3) 
# checkpoint name
ckpt_name = "segnet_resnet5.hdf5"
model = resnet50_segnet(n_classes=5, 
                        input_height=im_res_[1], 
                        input_width=im_res_[0])
print (model.summary())
# loads the weights from the checkpoint model 
model.load_weights(join("ckpt/", ckpt_name))


im_h, im_w = im_res_[1], im_res_[0]
def testGenerator():
    # test all images in the directory
    assert exists(test_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(test_dir):
        # read and scale inputs
        img = Image.open(p).resize((im_w, im_h))
        img = np.array(img)/255.
        # expand dimensions by 1 for matrix multiplication
        img = np.expand_dims(img, axis=0)
        # inference
        out_img = model.predict(img)
        # thresholding
        # if the prediction is greater than 0.5 is the value for the mask
        out_img[out_img>0.5] = 1.
        out_img[out_img<=0.5] = 0.
        print ("tested: {0}".format(p))
        # get filename
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        # save individual output masks
        ROs = np.reshape(out_img[0,:,:,0], (im_h, im_w))
        FVs = np.reshape(out_img[0,:,:,1], (im_h, im_w))
        HDs = np.reshape(out_img[0,:,:,2], (im_h, im_w))
        RIs = np.reshape(out_img[0,:,:,3], (im_h, im_w))
        WRs = np.reshape(out_img[0,:,:,4], (im_h, im_w))
        Image.fromarray(np.uint8(ROs*255.)).save(RO_dir+img_name)
        Image.fromarray(np.uint8(FVs*255.)).save(FB_dir+img_name)
        Image.fromarray(np.uint8(HDs*255.)).save(HD_dir+img_name)
        Image.fromarray(np.uint8(RIs*255.)).save(RI_dir+img_name)
        Image.fromarray(np.uint8(WRs*255.)).save(WR_dir+img_name)

# test images
testGenerator()



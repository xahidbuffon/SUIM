"""
# Training pipeline of the SUIM-Net
# Paper:  
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Usage: for academic and educational purposes only
"""
from __future__ import print_function, division
import os
# keras libs
from keras.callbacks import ModelCheckpoint
# local libs
from model import SUIM_Net
from utils.data_utils import trainDataGenerator

## dataset directory
dataset_name = "suim"
train_dir = "/mnt/data1/ImageSeg/suim/train_val/"

## ckpt directory
ckpt_dir = "ckpt/"
model_ckpt_name = os.path.join(ckpt_dir, "suim_net5.hdf5")
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)

## input/output shapes
im_w, im_h, chan = 320, 240, 3

# setup data generator
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

## initialize model
model = SUIM_Net(im_res=(im_h, im_w), n_classes=5).model
print (model.summary())

Load_ckpt = True
# load saved model
if Load_ckpt: model.load_weights(model_ckpt_name)
model_checkpoint = ModelCheckpoint(model_ckpt_name, 
                                   monitor = 'loss', 
                                   verbose = 1, 
                                   save_best_only = True)

# data generator
train_gen = trainDataGenerator(4, # batch_size 
                              train_dir,# train-data dir
                              "images", # image_folder 
                              "masks", # mask_folder
                              data_gen_args, # aug_dict
                              image_color_mode="rgb", 
                              mask_color_mode="rgb",
                              target_size = (im_h, im_w))

## fit model
model.fit_generator(train_gen, 
                    steps_per_epoch = 5000,
                    epochs = 50,
                    callbacks = [model_checkpoint])


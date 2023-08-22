"""
# Training pipeline of PSPNet (MobileNet) on SUIM 
"""
from __future__ import print_function, division
import os
from os.path import join, exists
from keras import callbacks
# local libs
from models.pspnet import mobilenet_pspnet
from utils.data_utils import trainDataGenerator

## dataset directory
dataset_name = "SUIM"
train_dir = "data/train_val/"

## ckpt directory
ckpt_dir = "myckpt/"
im_res_ = (384, 384, 3)
ckpt_name = "pspnet_mobilenet.hdf5"
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir): os.makedirs(ckpt_dir)

## initialize model
model = mobilenet_pspnet(n_classes=5, 
                        input_height=im_res_[1], 
                        input_width=im_res_[0])
print (model.summary())
## load saved model
#model.load_weights(join("ckpt/saved/", "***.hdf5"))


batch_size = 2
num_epochs = 20
# setup data generator
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model_checkpoint = callbacks.ModelCheckpoint(model_ckpt_name, 
                                   monitor = 'loss', 
                                   verbose = 1, mode= 'auto',
                                   save_weights_only = True,
                                   save_best_only = True)

# data generator
train_gen = trainDataGenerator(batch_size, # batch_size 
                              train_dir,# train-data dir
                              "images", # image_folder 
                              "masks", # mask_folder
                              data_gen_args, # aug_dict
                              image_color_mode="rgb", 
                              mask_color_mode="rgb",
                              target_size = (im_res_[1], im_res_[0]))

## fit model
model.fit(train_gen, 
          steps_per_epoch = 2,#000,
          epochs = num_epochs,
          callbacks = [model_checkpoint])


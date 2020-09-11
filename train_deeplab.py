from __future__ import print_function, division
import os
from os.path import join, exists

# keras libs
from keras import callbacks
# local libs
from models.deeplabv3 import Deeplabv3
from utils.data_utils import trainDataGenerator


## dataset and experiment directories
dataset_name = "suim"
data_dir = "/mnt/data1/ImageSeg/suim/traim_val/"
#data_dir = "./square/"
train_dir, val_dir = data_dir, data_dir # os.path.join(data_dir, "train"), os.path.join(data_dir, "test")


## ckpt directory
ckpt_dir = "ckpt/"
im_res_ = (320, 320, 3)
ckpt_name = "deeplabv3.hdf5"
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir): os.makedirs(ckpt_dir)

## initialize model
model = Deeplabv3(weights=None, input_shape=im_res_, classes=5)
print(model.summary())
## load saved model
#model.load_weights(join("ckpt/saved/", "***.hdf5"))

batch_size = 2
num_epochs = 50

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
                                   verbose = 1, 
                                   save_best_only = True)



# data generator
train_gen = trainDataGenerator(batch_size, # batch_size
                              train_dir,# train-data dir
                              "images", # image_folder
                              "masks", # mask_folder
                              data_gen_args, # aug_dict
                              image_color_mode="rgb", 
                              mask_color_mode="rgb", 
                              save_to_dir = None, 
                              target_size = (im_res_[1], im_res_[0]))

## fit model
model.fit_generator(train_gen, 
                    steps_per_epoch = 4000,
                    epochs = num_epochs,
                    callbacks = [model_checkpoint])


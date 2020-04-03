"""
# Data utility functions for training on the SUIM dataset
# Paper:  
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Usage: for academic and educational purposes only
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import fnmatch
import itertools as it
from scipy import misc

"""
RGB color code and object categories:
------------------------------------
000 BW: Background waterbody
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)
"""
def getRobotFishHumanReefWrecks(mask):
    # for 5 categories: human, robot, fish, wrecks, reefs
    imw, imh = mask.shape[0], mask.shape[1]
    Human = np.zeros((imw, imh))
    Robot = np.zeros((imw, imh))
    Fish = np.zeros((imw, imh))
    Reef = np.zeros((imw, imh))
    Wreck = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
                Human[i, j] = 1 
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
                Robot[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                Fish[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1):
                Reef[i, j] = 1  
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
                Wreck[i, j] = 1  
            else: pass
    return np.stack((Robot, Fish, Human, Reef, Wreck), -1) 


def getSaliency(mask):
    # for 4 categories: human, robot, fish, wrecks
    imw, imh = mask.shape[0], mask.shape[1]
    Human = np.zeros((imw, imh))
    Robot = np.zeros((imw, imh))
    Fish = np.zeros((imw, imh))
    Wreck = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
                Human[i, j] = 1 
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
                Robot[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                Fish[i, j] = 1  
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
                Wreck[i, j] = 1  
            else: pass
    return np.stack((Robot, Fish, Human, Wreck), -1) 


def processSUIMDataRFHW(img, mask, sal=False):
    # scaling image data and masks
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    m = []
    for i in range(mask.shape[0]):
        if sal:
            m.append(getSaliency(mask[i]))
        else:
            m.append(getRobotFishHumanReefWrecks(mask[i]))
    m = np.array(m)
    return (img, m)


def trainDataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", target_size=(256,256), sal=False):
    # data generator function for driving the training
    image_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = "image",
        seed=1)
    # mask generator function for corresponding ground truth
    mask_datagen = ImageDataGenerator(**aug_dict)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = "mask",
        seed = 1)
    # make pairs and return
    for (img, mask) in it.izip(image_generator, mask_generator):
        img, mask_indiv = processSUIMDataRFHW(img, mask, sal)
        yield (img, mask_indiv)


def getPaths(data_dir):
    # read image files from directory
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG', '*.bmp']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return image_paths


def read_and_resize(path, img_res):
    # read and resize image files
    img = misc.imread(path, mode='RGB').astype(np.float)  
    img = misc.imresize(img, img_res)
    return img


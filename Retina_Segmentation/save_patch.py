# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza Azad
"""
from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from help_functions import *
from extract_patches import *
import argparse

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training

#========= Load settings from Config file
#patch to the datasets
path_data = './DRIVE_datasets_training_testing/'

parser = argparse.ArgumentParser()
parser.add_argument('--patch_size', type=int, default=64,
                    help='Patch size')
parser.add_argument('--n_patches', type=int, default=200000,
                    help='Total number of patches')
args = parser.parse_args()

print('extracting patches')
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + 'DRIVE_dataset_imgs_train.hdf5',
    DRIVE_train_groudTruth    = path_data + 'DRIVE_dataset_groundTruth_train.hdf5',  #masks
    patch_height = args.patch_size,
    patch_width  = args.patch_size,
    N_subimgs    = args.n_patches,
    inside_FOV = 'True' #select the patches only inside the FOV  (default == True)
)


#np.save('patches_imgs_train',patches_imgs_train)
#np.save('patches_masks_train',patches_masks_train)



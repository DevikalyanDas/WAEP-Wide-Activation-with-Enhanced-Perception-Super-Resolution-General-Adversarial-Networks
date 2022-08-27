import sys
sys.path.insert(1, '../code/test/code_waa')

import os
import numpy as np
import cv2
import random
from pathlib import Path
import pandas as pd

import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from PIL import Image

from statistics import mean
from math import log10
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from transforms import *

class custom_test_dataset(data.Dataset):

    def __init__(self,img_dir,lb_fol,img_size=(286,286),img_index=0):
        super(custom_test_dataset, self).__init__()
        self.img_dir = img_dir
        self.lb_fol = lb_fol
        self.img_index = img_index
        self.index_1 = -1
        self.img_size = img_size

        tf_test_list =[
                      ToTensor(),
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ]

        self.tf_test = Compose(tf_test_list)
        print("Found %d test images" % (len(self.img_index)))

    def __len__(self):
        return 1500

    def __getitem__(self,index):

        # 16 patches for one image index
        # index_1 -> for each patch, index_2-> for each image containing 16 patches
        if(index%15==0):
          self.index_1 += 1
        index_2 = index%15

        # Read Image
        patch_path = os.path.join(self.img_dir, self.lb_fol[0], self.img_index[self.index_1] + 'x4')
        patch_index = next(os.walk(patch_path))[2]
        patch_index = sorted(patch_index, key = lambda x: (os.path.splitext(x)[0]))
        img_path = os.path.join(patch_path, patch_index[index_2])
        img = Image.open(img_path)

        # Read Label (HR)
        patch_path_hr = os.path.join(self.img_dir, self.lb_fol[1], self.img_index[self.index_1])
        patch_index_hr = next(os.walk(patch_path_hr))[2]
        patch_index_hr = sorted(patch_index_hr, key=lambda x: (os.path.splitext(x)[0]))
        lbl_path = os.path.join(patch_path_hr, patch_index_hr[index_2])
        lbl = Image.open(lbl_path)

        # transforms
        img, lbl = self.tf_test(img, lbl)
        return img, lbl, self.img_index[self.index_1]

# ======================================================================================
# SANITY CHECK FOR TEST DATALOADER - Run only when dataloader is to be verified
# ======================================================================================

# ======================================================================================
# Directories to local files
# ======================================================================================
# path_imgs = '/content/gdrive/MyDrive/HLCV/HLCV_project/DIV2K_HR/train_valid_HR'
# test_hr = '/content/gdrive/MyDrive/HLCV/HLCV_project/DIV2K_HR/test_HR'
# local_path = "/content/gdrive/MyDrive/HLCV/HLCV_project/test"
# test_fol = ['patch_lr','patch_hr']

# # csv_dir = '/content/gdrive/MyDrive/HLCV/HLCV_project/csv_output'
# # saved_weights_dir = '/content/gdrive/MyDrive/HLCV/HLCV_project/saved_weights'
# # output_dir = '/content/gdrive/MyDrive/HLCV/HLCV_project/train_output'
# # test_patch_lr = '/content/gdrive/MyDrive/HLCV/HLCV_project/test/patch_lr'
# # test_patch_hr = '/content/gdrive/MyDrive/HLCV/HLCV_project/test/patch_hr'
# #======================================================================================
# # Hyper-parameters
# #======================================================================================
# batch_size_test = 1
# #======================================================================================
# # Load dataset
# #======================================================================================
# img_ind = next(os.walk(test_patch_hr))[1]
# img_ind = sorted(img_ind, key = lambda x: (os.path.splitext(x)[0]))

# # Test set
# dst_test = custom_test_dataset(img_dir = local_path,
#                           img_size = (64,64),
#                           lb_fol = test_fol,
#                           img_index = img_ind)

# test_loader = DataLoader(dst_test,
#                           batch_size = batch_size_test,
#                           num_workers = 0,
#                           shuffle=False)

# plotting the test set:
# for i, data_samples in enumerate(test_loader):
#     imgs, labels, file_name = data_samples
#     print(file_name[0])
#     imgs = imgs.permute(0,2,3,1)
#     labels = labels.permute(0,2,3,1)

#     imgs = imgs.numpy().squeeze()
#     labels = labels.numpy().squeeze()

#     f, ax1 = plt.subplots(1, 2,figsize=(14,14))
#     ax1[0].set_title('LR Image')
#     ax1[1].set_title('SR Image')

#     ax1[0].imshow(imgs)
#     ax1[1].imshow(labels)

#     f.tight_layout()
#     plt.show()

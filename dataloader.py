import os
import numpy as np
import cv2
import random
from pathlib import Path
import pandas as pd

import torch
import torchvision
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

class custom_dataset(data.Dataset):

    def __init__(self,img_dir,lb_fol,data_list,mode='Train',img_size=(286,286)):
        super(custom_dataset, self).__init__()
        self.img_dir = img_dir
        self.lb_fol = lb_fol

        self.mode = mode
        self.image_filenames = data_list
        self.img_size = img_size

        transforms_list = [
                           RandomHorizontalFlip(0.5),
                           RandomVerticalFlip(0.5),
                           RandomRotate(degree=0),
                          ]

        tf_test_list =[
                CustomRandomCrop(64),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ] #64x64 cropping

        self.tf_train = Compose(transforms_list + tf_test_list)
        self.tf_test = Compose(tf_test_list)

        print("Found %d %s images" % (len(self.image_filenames), mode))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,index):
        # Read Image
        img_path = os.path.join(self.img_dir,self.lb_fol[1],'x4',self.lb_fol[3],self.image_filenames[index].split('.')[0]+'x4.png')
        img = cv2.imread(img_path)
#         print(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.uint8)

        # Read GT
        lbl_path = os.path.join(self.img_dir,self.lb_fol[0],self.lb_fol[2],self.image_filenames[index])
#         print(lbl_path)
        lbl = cv2.imread(lbl_path)
        lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
        lbl = np.array(lbl, dtype=np.uint8)

        img = Image.fromarray(img.astype('uint8'), 'RGB')
        lbl = Image.fromarray(lbl.astype('uint8'), 'RGB')
        # Transforms
        if self.mode == 'Train':
            img, lbl = self.tf_train(img, lbl)
        if self.mode != 'Train':
            img, lbl = self.tf_test(img, lbl)

        return img, lbl, self.image_filenames[index]


"""Sanity check for dataloader"""
# batch_size = 4
# batch_size_val = 4

# path_imgs = '/content/gdrive/MyDrive/HLCV/HLCV_project/DIV2K_HR/train_valid_HR'
# test_hr = '/content/gdrive/MyDrive/HLCV/HLCV_project/DIV2K_HR/test_HR'
# local_path = "/content/gdrive/MyDrive/HLCV/HLCV_project/"
# train_fol = ['DIV2K_HR','DIV2K_LR','train_valid_HR','train_valid_x4']
# valid_fol = ['DIV2K_HR','DIV2K_LR','train_valid_HR','train_valid_x4']
# test_fol = ['DIV2K_HR','DIV2K_LR','test_HR','test_x4']

# img_ind = next(os.walk(path_imgs))[2]
# test_data = next(os.walk(test_hr))[2]

# random.shuffle(img_ind)
# train_data = img_ind[:int((len(img_ind)+1)*.80)] # 80% to training set
# valid_data = img_ind[int((len(img_ind)+1)*.80):] # 20% to validation set

# # Training set
# dst_train = custom_dataset(img_dir = local_path,
#                             mode = "Train",
#                             data_list = train_data,
#                             img_size = (256,256),
#                             lb_fol = train_fol)

# train_loader = DataLoader(dst_train,
#                           batch_size = batch_size,
#                           num_workers = 0,
#                           shuffle = True)

# #plotting the train set:
# for i, data_samples in enumerate(train_loader):

#     imgs, labels, ids = data_samples

#     imgs = denormalize(imgs).detach().cpu().permute(0,2,3,1).numpy().squeeze()
#     labels = denormalize(labels).detach().cpu().permute(0,2,3,1).numpy().squeeze()

#     f, ax1 = plt.subplots(4, 2,figsize=(14,14))
#     ax1[0][0].set_title('LR Image')
#     ax1[0][1].set_title('SR Image')

#     for j in range(batch_size):
#         ax1[j][0].imshow(imgs[j])
#         ax1[j][1].imshow(labels[j])

#     f.tight_layout()
#     plt.show()
#     break

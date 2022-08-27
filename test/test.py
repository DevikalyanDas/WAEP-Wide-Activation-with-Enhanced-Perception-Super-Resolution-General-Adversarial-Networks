#!/usr/bin/env python3
import sys
#sys.path.insert(1, '/netscratch/devidas/hlcv3/code/test/code_waa')
import os
import numpy as np
import pandas as pd
import random
import warnings
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision import datasets, models, transforms

from PIL import Image
import PIL
import cv2
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project libraries
from model import *
from utils import * # metricsCalculator, psnr, save_history
# import ssim
from test_dataloader import custom_test_dataset
from transforms import *
from test_utils import join_patch

# To supress the warnings at the output
warnings.filterwarnings("ignore")
# Empty the cache for CUDA memory
torch.cuda.empty_cache()
#======================================================================================
# Device configuration
#======================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)
#======================================================================================
# Directories to local files
#======================================================================================
path_imgs = '../DIV2K_LR/x4/test_x4'
test_hr = '../DIV2K_HR/test_HR'
local_path = "../test"
test_fol = ['patch_lr','patch_hr']

csv_dir = '../csv_output'
saved_weights_dir = '../saved_weights'
output_dir = '../train_output'

test_patch_lr = '../test/patch_lr'
test_patch_hr = '/netscratch/devidas/hlcv3/test/patch_hr'

test_input_dir = '../test/test_img_lr'
test_output_dir = '../test/test_gen_hr'
test_label_dir = '../test/test_label_hr'
test_ip_lbl_compare_dir = '../test/test_compare_output'

create_dir(test_patch_lr)
create_dir(test_patch_hr)
create_dir(test_input_dir)
create_dir(test_output_dir)
create_dir(test_label_dir)
create_dir(test_ip_lbl_compare_dir)
#create_dir(test_metrics)
#======================================================================================
# Hyper-parameters
#======================================================================================
batch_size_test = 1

lambda_adv = 5e-3
lambda_pixel = 1e-2
residual_blocks = 18 #by default
filters_G = 128 #by default
b1 = 0.9
b2 = 0.99
lr_size = 64
hr_size = 256
#======================================================================================
# Load dataset
#======================================================================================
img_ind = next(os.walk(test_patch_hr))[1]
img_ind = sorted(img_ind, key = lambda x: (os.path.splitext(x)[0]))

# Test set
dst_test = custom_test_dataset(img_dir = local_path,
                          img_size = (64,64),
                          lb_fol = test_fol,
                          img_index = img_ind)

test_loader = DataLoader(dst_test,
                          batch_size = batch_size_test,
                          num_workers = 0,
                          shuffle=False)
#======================================================================================
# Test loop definition
#======================================================================================
saved_weights = torch.load(os.path.join(saved_weights_dir,'model_61.pt'), map_location=torch.device('cpu'))

with torch.no_grad():
  generator = Generator(3, filters=filters_G, num_res_blocks=residual_blocks).to(device)
  generator.eval()
  generator.load_state_dict(saved_weights)

  output_patch_list = []
  input_patch_list = []
  label_patch_list = []

  for iteration, samples in tqdm(enumerate(test_loader), total = len(test_loader)):
      imgs_lr, imgs_hr, file_name = samples

      imgs_lr = imgs_lr.to(device)
      imgs_hr = imgs_hr.to(device)

      # Generate a high resolution image from low resolution input
      gen_hr = generator(imgs_lr)

      imgs_lr = denormalize(imgs_lr).detach().cpu().permute(0,2,3,1).numpy().squeeze()
      imgs_hr = denormalize(imgs_hr).detach().cpu().permute(0,2,3,1).numpy().squeeze()
      gen_hr  = denormalize(gen_hr).detach().cpu().permute(0,2,3,1).numpy().squeeze()

      output_patch_list.append(gen_hr)
      input_patch_list.append(imgs_lr)
      label_patch_list.append(imgs_hr)

      # Once first 15 images have been iterated, time to join them to a single image
      if (len(output_patch_list)==15):
        final_input  = join_patch(input_patch_list, "lr")
        final_output  = join_patch(output_patch_list, "hr")
        final_label  = join_patch(label_patch_list, "hr")

        final_input = (final_input*255).astype(np.uint8)
        final_output = (final_output*255).astype(np.uint8)
        final_label = (final_label*255).astype(np.uint8)

        f, ax1 = plt.subplots(3, figsize=(14,14))
        ax1[0].set_title('LR Image')
        ax1[1].set_title('Predicted HR Image')
        ax1[2].set_title('Original HR Image')

        ax1[0].imshow(final_input)
        ax1[1].imshow(final_output)
        ax1[2].imshow(final_label)
        f.tight_layout()
        plt.savefig(os.path.join(test_ip_lbl_compare_dir, 'fig_{}.png'.format(file_name[0])), dpi=500)

        # Perform this operation after saving the above figure - bcz of cv2
        final_input = cv2.cvtColor(final_input, cv2.COLOR_BGR2RGB)
        final_output = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        final_label = cv2.cvtColor(final_label, cv2.COLOR_BGR2RGB)

        # Join patches and save them in respective directories
        # Patches for labels have been already been joined while creating the patches (refer create_patch)
        cv2.imwrite(os.path.join(test_input_dir, file_name[0] + '.png'), final_input)
        cv2.imwrite(os.path.join(test_output_dir, file_name[0] + '.png'), final_output)
        cv2.imwrite(os.path.join(test_label_dir, file_name[0] + '.png'), final_label)

        output_patch_list = []
        input_patch_list = []
        label_patch_list = []
        #print("Image {}.png saved".format(file_name[0]))

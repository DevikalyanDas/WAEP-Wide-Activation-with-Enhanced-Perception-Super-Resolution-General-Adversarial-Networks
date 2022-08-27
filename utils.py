import os
import torch
import numpy as np
import pandas as pd
import math
import torch.nn as nn
import ssim
import PIL.Image as pil_image
import torchvision.transforms as transforms
from transforms import *
#======================================================================================
# create directories
#======================================================================================
def create_dir(path):
    dir = os.path.join(path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    # else :
    #   os.remove(dir)

#======================================================================================
# For avging metrics values
#======================================================================================
class MetricsCalculator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#======================================================================================
# Calculates psnr batch wise
#======================================================================================
def psnr(original, compressed): 
    running_psnr = 0
    for original_idx, compressed_idx in zip(original, compressed):
        # original_idx = 255*original_idx
        # compressed_idx = 255*compressed_idx
        # original_idx = torch.clamp(original_idx, min=0, max=255)
        # compressed_idx = torch.clamp(compressed_idx, min=0, max=255)
        
        original_idx = denormalize(original_idx.detach())*255
        compressed_idx = denormalize(compressed_idx.detach())*255
        mse = torch.mean((original_idx - compressed_idx) ** 2) 
        #print(mse)
        if(mse == 0):  # MSE zero-> No noise is present-> PSNR has no importance. 
            return 100
        max_pixel_val = 255.0
        running_psnr += 20 * torch.log10(max_pixel_val / torch.sqrt(mse))
    psnr = running_psnr / (original.shape)[0]
    return psnr

#======================================================================================
# Copy all output data to CSV files. Can be used for plotting later on..
#======================================================================================
def save_history(loss_data, psnr_data, ssim_data, path_dir): 
    create_dir(path_dir)
    pd.DataFrame.from_dict(data=loss_data, orient='columns').to_csv(os.path.join(path_dir,'loss.csv'), header=['epoch', 'lr', 'train loss','val loss'])
    pd.DataFrame.from_dict(data=psnr_data, orient='columns').to_csv(os.path.join(path_dir,'psnr.csv'), header=['epoch','train psnr','val psnr'])
    pd.DataFrame.from_dict(data=ssim_data, orient='columns').to_csv(os.path.join(path_dir,'ssim.csv'), header=['epoch','train ssim','val ssim'])

#======================================================================================
# Weighted loss function - SSIM, PSNR, MSE
#======================================================================================
def weighted_loss(original,compressed):
    '''
    Original and compressed are torch tensors
    0.4 = MSE
    0.5 = PSNR
    0.1 = SSIM
    '''
    mseLoss = nn.MSELoss()
    mse = mseLoss(original, compressed)
    
    psnr_score = psnr(original, compressed) 
    #PSNR is maximized so 100-PSNR is a loss function (Or -PSNR)
    psnr_loss = 100 - psnr_score 

    ssim_score = ssim.ssim(original,compressed)
    ssim_loss = 1 - ssim_score.item()

    weighted_loss = (0.4 * mse) + (0.5 * (psnr_loss/100)) + (0.1 * ssim_loss)
    return weighted_loss

#======================================================================================
# Fetch gaussian variance which is linearly decresing till 6K iterations from 1-->0
#======================================================================================
def fetch_gauss_variance(iteration):
    m = -1/5999
    c = 6000/5999
    variance = m*(iteration+1) + c
    return variance

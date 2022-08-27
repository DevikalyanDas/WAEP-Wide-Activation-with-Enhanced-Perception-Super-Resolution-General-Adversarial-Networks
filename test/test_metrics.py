import sys
sys.path.insert(1, '../code/')
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

# Imports all metrics libraries
# Run following commands before running the code
# pip install lpips
# pip install image-quality
from utils import * #ssim
import lpips #lpips
import imquality.brisque as brisque #brisque

test_output_dir = '../code/test/test_gen_hr'
test_label_dir = '../code/test/test_label_hr'
test_path = '../code/test/'

#======================================================================================
# Device configuration
#======================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

# Fetch the image names (0801.png, 0802.png etc)
img_ind = next(os.walk(test_label_dir))[2]
#======================================================================================
# PSNR for test data - without de-normalisation
#======================================================================================
def testdata_psnr(original, compressed):
    running_psnr = 0
    for original_idx, compressed_idx in zip(original, compressed):
        mse = torch.mean((original_idx - compressed_idx) ** 2)
        if(mse == 0):  # MSE zero-> No noise is present-> PSNR has no importance.
            return 100
        max_pixel_val = 255.0
        running_psnr += 20 * torch.log10(max_pixel_val / torch.sqrt(mse))
    psnr = running_psnr / (original.shape)[0]
    return psnr

metrics = {"ImageID": [], "PSNR":[], "SSIM": [], "LPIPS": [], "BRISQUE": []}

running_psnr = 0
running_ssim = 0
running_lpips = 0
running_brisque = 0

# Create an instance of LPIPS class and use it in the loop below
get_lpips = lpips.LPIPS(net='alex')

for file_num, image_idx in enumerate(img_ind):
  pred = cv2.imread(os.path.join(test_output_dir, image_idx))
  label = cv2.imread(os.path.join(test_label_dir, image_idx))

  pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB).astype(np.float32)
  label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB).astype(np.float32)

  #BRISQUE - works on numpy array
  brisque_score = brisque.score(pred)

  #PSNR - need to convert images to tensor
  pred = torch.from_numpy(pred)
  label = torch.from_numpy(label)

  pred = torch.Tensor(pred)
  label = torch.Tensor(label)

  psnr_score = testdata_psnr(pred, label)

  #SSIM - needs a 4d tensor
  pred = pred.unsqueeze(0)
  label = label.unsqueeze(0)
  ssim_score = ssim.ssim(pred,label)

  #LPIPS - change the order of the channels
  pred = pred.permute(0,3,1,2)
  label = label.permute(0,3,1,2)
  lpips_score = get_lpips(pred, label)

  running_psnr += psnr_score
  running_ssim += ssim_score
  running_lpips += lpips_score
  running_brisque += brisque_score
  print("File {}, PSNR {}, SSIM {}, LPIPS {}, BRISQUE {}".format(file_num, psnr_score.item(), ssim_score.item(), lpips_score.item(), brisque_score))

  metrics["ImageID"].append(image_idx)
  metrics["SSIM"].append(ssim_score.item())
  metrics["LPIPS"].append(lpips_score.item())
  metrics["BRISQUE"].append(brisque_score)

  try : # if MSE is zero, PSNR returns a scalar value of 100
    metrics["PSNR"].append(psnr_score.item())
  except AttributeError:
    metrics["PSNR"].append(psnr_score)

# Compute avg values
avg_psnr = running_psnr/len(img_ind)
avg_ssim = running_ssim/len(img_ind)
avg_lpips = running_lpips/len(img_ind)
avg_brisque = running_brisque/len(img_ind)

metrics["ImageID"].append("Average Scores")
metrics["PSNR"].append(avg_psnr)
metrics["SSIM"].append(avg_ssim)
metrics["LPIPS"].append(avg_lpips)
metrics["BRISQUE"].append(avg_brisque)

print('-'*10)
print("Avg PSNR : ", avg_psnr)
print("Avg SSIM : ", avg_ssim)
print("Avg LPIPS : ", avg_lpips)
print("Avg BRISQUE : ", avg_brisque)
pd.DataFrame.from_dict(data=metrics, orient='columns').to_csv(os.path.join(test_path,'test_metrics.csv'), header=['ImageID','PSNR','SSIM','LPIPS','BRISQUE'])

import os
import cv2
import numpy as np
from tqdm import tqdm

local_path = "../"
test_fol = ['DIV2K_HR','DIV2K_LR','test_HR','test_x4']
test_folder = '../test'
test_patch_lr = '../test/patch_lr'
test_patch_hr = '../test/patch_hr'
test_label_dir = '../test/test_label_hr'

# create directories
def create_dir(path):
    dir = os.path.join(path)
    if not os.path.exists(dir):
        os.mkdir(dir)
        #print("created {}".format(dir))

create_dir(test_folder)
create_dir(test_patch_lr)
create_dir(test_patch_hr)
create_dir(test_label_dir)
#======================================================================================
# Finds minimum dimensions of the images in the dataset (LR and HR images both)
#======================================================================================
def find_min_dim():
  x_dim = []
  y_dim = []
  idx_ip = next(os.walk(os.path.join(local_path, test_fol[1], 'x4', test_fol[3])))[2]
  for iterate in idx_ip :
    img = cv2.imread(os.path.join(local_path, test_fol[1], 'x4', test_fol[3], iterate))
    x_dim.append(img.shape[0])
    y_dim.append(img.shape[1])

  print(min(x_dim), min(y_dim))
#======================================================================================
# Divide a single image into 15 patches (3x5 grid)
#======================================================================================
def divide_patch(img, img_type):

  if (img_type=="lr"):
    x_dim = 192 # minimum height of the LR images
    y_dim = 320 # minimum width of the HR image
    x_stride = x_dim//64
    y_stride = y_dim//64
    patch_height = 64
    patch_width = 64

  else:
    x_dim = 768 # minimum height of HR image
    y_dim = 1280 # minimum width of HR image
    x_stride = x_dim//256 #3
    y_stride = y_dim//256 #5
    patch_height = 256
    patch_width = 256

  patch_list = []
  for i in range(x_stride):
    for j in range(y_stride):
      patch_img = img[i*patch_width : i*patch_width+patch_width,
                      j*patch_height : j*patch_height+patch_height]
      patch_list.append(patch_img)

  return patch_list

#======================================================================================
# Save the divided patches into a list and then to a directory for future use in test loop
#======================================================================================
def create_patch(img_type):

  if (img_type == "lr"):
    path = os.path.join(local_path, test_fol[1], 'x4', test_fol[3])
    img_size = [192,320]
  else:
    path = os.path.join(local_path, test_fol[0], test_fol[2])
    img_size = [768,1280]

  idx_ip = next(os.walk(path))[2]

  for iterate in idx_ip :
    img = cv2.imread(os.path.join(path, iterate))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(img_size[1], img_size[0]), interpolation = cv2.INTER_CUBIC)
    patches = divide_patch(img, img_type)

    if (img_type=="hr"):
      # If the mode is HR, create the patches and then join them too. Will be used later during test loop
      final_label_img = join_patch(patches, "hr")
      file_name = '%s/%s.png'%(test_label_dir, iterate.split(".")[0])
      cv2.imwrite(file_name, final_label_img)

    if (img_type=="lr"):
      path_patch = os.path.join(test_patch_lr, iterate.split(".")[0])
    else :
      path_patch = os.path.join(test_patch_hr, iterate.split(".")[0])

    create_dir(path_patch)
    for patch_idx in range(len(patches)):
      patch_idx_rename = "{0:02d}".format(patch_idx)
      patch_file = '%s/%s_%s.png'%(path_patch, iterate.split(".")[0], patch_idx_rename)
      cv2.imwrite(patch_file, patches[patch_idx])

  print("Patch creation done !")

#======================================================================================
# Merge the patches to a single image
#======================================================================================
def join_patch(patch_list, img_type):

  if (img_type == "lr"):
    img_size = [192,320]
    x_stride = img_size[0]//64 #3
    y_stride = img_size[1]//64 #5
    patch_height = 64
    patch_width = 64
    idx_ip = next(os.walk(test_patch_lr))[1]

  else:
    img_size = [768,1280]
    x_stride = img_size[0]//256 #3
    y_stride = img_size[1]//256 #5
    patch_height = 256
    patch_width = 256
    idx_ip = next(os.walk(test_patch_hr))[1]

  final_img = np.zeros((img_size[0], img_size[1], 3))
  count = 0
  for i in range(x_stride):
    for j in range(y_stride):
      final_img[i*patch_width : i*patch_width+patch_width,
                j*patch_height : j*patch_height+patch_height] = patch_list[count]
      count+=1
  return final_img

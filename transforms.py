import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

################## custom transform  ##################

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Resize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        
        return np.asarray(image), np.asarray(target)


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.choice([0,180,90,270])
        return (
            F.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=(0, 0, 0),
                shear=0.0,
            ),
            F.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=250,
                shear=0.0,
            ),
        )

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = F.normalize(target, mean=self.mean, std=self.std)
        return image, target
 
 
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CustomRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.size, self.size))
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i*4, j*4, h*4, w*4)
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        image = F.adjust_brightness(image, self.brightness)
        image = F.adjust_contrast(image, self.contrast)
        image = F.adjust_saturation(image, self.saturation)
        image = F.adjust_hue(image, self.hue)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target        


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms[:-2]
        self.compulsory = transforms[-2:]
    def __call__(self, image,target):

        tr = random.choice(self.transforms)
        # img_1, tgt_1 = tr(image,target)
        trans = self.compulsory+[tr]
        for t in trans:
            image, target = t(image, target)
        return image,target  

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 1.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean
    
    # def __repr__(self):
    #     return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

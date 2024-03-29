import os
import glob
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
from utils import *
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, img_dir, dir_info=None, rand_map=True , scale=True, patch=None, max_len=None):
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

        if dir_info:
            img_path = os.path.join(img_dir, dir_info[0], f"{dir_info[1]}")
            print(f"Image path: {img_path}")
            noisy_files = glob.glob(img_path)
        else:
            noisy_files = glob.glob(os.path.join(img_dir, "hdr", "*.jpg"))
            #noisy_files += glob.glob(os.path.join(img_dir, "data", "*01_0.1s_3200.jpg"))
        print(f"Found {len(noisy_files)} noisy files.")
        assert len(noisy_files) != 0


        self.train_low_data_names = noisy_files
        self.train_low_data_names.sort()      

        self.rand_map = rand_map

        self.scale = scale
        self.patch = False
        if patch:
            self.patch = True
        
        if max_len:
            self.train_low_data_names = self.train_low_data_names[:max_len]
        

    def load_images_transform(self, file):
        im = np.asarray(cv2.imread(file))
        if self.scale:
            im = cv2.resize(im, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        im_h, im_w, _ = im.shape
        im_h = (im_h//8)*8
        im_w = (im_w//8)*8
        img = im[:im_h, :im_w, :]
        if self.patch:
            patch = 512

            h_offset = random.randint(0, max(0, im_h - patch - 1))
            w_offset = random.randint(0, max(0, im_w - patch - 1))

            img = img[h_offset:h_offset + patch, w_offset:w_offset + patch, :]
        # if self.rand_map:
        #     rand_map = get_random_expmap(img.shape, patch_size=64)
        #     img = np.concatenate((img, rand_map), axis=2)
        # else:
        #     exp_map = get_expmap(img, s = 0.55)
        #     img = np.concatenate((img, exp_map), axis=2)

        img[:, :, :3] = img[:, :, :3]/255.
        img = img2tensor(img, bgr2rgb=True)

        return img
        
    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])

        img_name = self.train_low_data_names[index].split('/')[-1]
        
        #normalize(low, self.mean, self.std, inplace=True)

        return low, img_name

    def __len__(self):
        return len(self.train_low_data_names)
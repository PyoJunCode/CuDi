import math
import random

import torch
import cv2
import numpy as np
from kornia.color import rgb_to_yuv

def get_expmap(img_tensor, s=0.55, a=0.15):
    if len(img_tensor.shape) != 4:
        raise Exception(f"Invalid Tensor dimension: 4 expected but got {len(img_tensor.shape)} instead.")
    img = rgb_to_yuv(img_tensor)

    mean = torch.mean(img[:, 0, :, :])
    
    img = torch.sub(mean, img[:, 0, :, :])
    
    # Normalize to [-1, 1]
    img = (torch.sub(img, torch.min(img))) / torch.sub(torch.max(img), torch.min(img))
    img = (2 * img) - 1
    
    # S + A * x
    exp_map = s + (img * a)
    
    exp_map = exp_map.unsqueeze(1)
    
    return exp_map

def get_random_expmap(img, exp_range=(0.2, 0.8), patch_size=256):
    if len(img.shape) != 4:
        raise Exception(f"Invalid Tensor dimension: 4 expected but got {len(img.shape)} instead.")
    
    bs, _, h, w = img.shape
    exp = torch.FloatTensor(bs, 1).uniform_(exp_range[0], exp_range[1]).to(img.device)
    
    exp_map = exp.view(bs, 1, 1, 1)
    
    exp_map = exp_map.repeat(1, 1, h, w)
    
    # Not Optimized.
    for emap in exp_map:
        # Patch size range = [patch size, path size *4]
        patch_size_log = int(math.log2(patch_size))
        patch = pow(2, random.randint(patch_size_log, patch_size_log+2))
        
        h_offset = random.randint(0, max(0, h - patch - 1))
        w_offset = random.randint(0, max(0, w - patch - 1))
        
        diff_exp = round(random.uniform(exp_range[0], exp_range[1]), 2)
        
        emap[:, h_offset:h_offset+patch, w_offset:w_offset+patch] = diff_exp
    
    
    return exp_map
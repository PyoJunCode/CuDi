import os
from collections import OrderedDict
import time

import torch
import pytorch_lightning as pl

from utils import *
from arch.led_arch import LEDNet
from losses import *

from kornia.losses import total_variation

class TeacherModel(pl.LightningModule):
    def __init__(self, save_path=None, lr=10e-4, use_side=False):
        super().__init__()
        self.model = LEDNet(connection=True)
        # Init model weight in normal distribution
        self.save_path = save_path

        self.use_side = use_side
        
        self.lr = lr

        # Losses
        # Self supervised spatial exposure control loss
        self.cri_side = L1Loss(loss_weight=0.8)
        self.cri_pix = L1Loss(loss_weight=1)
        self.cri_vgg = PerceptualLoss(
            {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
            perceptual_weight=0.01,
            range_norm=True,
            criterion="l1"

        )
        self.cri_is = IlluSmoothLoss(loss_weight=200)
    
        self.cri_cc = ColorLoss(loss_weight=5)
        # self.cri_sec = SpaExpLoss(loss_weight=1)
        # self.cri_sc = SpatialLoss(loss_weight=0.1)

    def training_step(self, batch, batch_idx):
        x, gt, image_name = batch
        
        l_total = 0
        losses = OrderedDict()
        
        # Forward
        side, res, r = self.model(x, side_loss=True)
        if self.use_side:
            side_h,side_w = side.shape[2:]
            side_gt = torch.nn.functional.interpolate(gt, (side_h, side_w), mode='bicubic', align_corners=False)
            
            l_side = self.cri_side(side, side_gt)
            l_total += l_side
            losses["l_side"] = l_side

        l_pix = self.cri_pix(res, gt)
        l_total += l_pix
        losses["l_pix"] = l_pix

        l_vgg, _ = self.cri_vgg(res, gt)
        l_total += l_vgg
        losses["l_vgg"] = l_vgg

        # l_sec = self.cri_sec(res, gt)
        # l_total += l_sec
        # losses["l_sec"] = l_sec  

        l_cc = self.cri_cc(res)
        l_total += l_cc
        losses["l_cc"] = l_cc

        # l_is = 200 * torch.mean(total_variation(r, reduction="mean"))
        # l_total += l_is
        # losses["l_is"] = l_is

        # l_sc = self.cri_sc(res, gt)
        # l_total += l_sc
        # losses["l_sc"] = l_sc


        losses["l_total"] = l_total


        self.log_dict(losses)
        return l_total


    def validation_step(self, batch, batch_idx):
        x, gt, image_names = batch
        res, r = self.model(x)

        return {"res": res, "image_names": image_names}

    def validation_epoch_end(self, outputs):
        for output in outputs:
            for res, image_name in zip(output["res"], output["image_names"]):
                # u_name = f'{self.current_epoch}/{image_name}_res_map.png'
                # u_path = os.path.join(self.save_path, u_name)
                # save_images(res_map, u_path)
                u_name = f'{self.current_epoch}/{image_name}_res.png'
                u_path = os.path.join(self.save_path, u_name)
                save_images(res, u_path, min_max=(0, 1))
            

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, _ = batch
        start = time.perf_counter()
        res = self.model(x)
        inf_time = time.perf_counter() - start
        
        return inf_time

    def test_epoch_end(self, outputs):
        avg = sum(outputs[1:]) / (len(outputs) - 1)
        print(f"Average inference time: {avg}")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, image_name = batch
        image_name = image_name[0].split('\\')[-1].split('.')[0]
        res, _ = self.model(x)

        u_name = f'{self.current_epoch}/{image_name}_inference.png'
        u_path = os.path.join(self.save_path, u_name)
        save_images(res, u_path, min_max=(0, 1))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        
        return optimizer
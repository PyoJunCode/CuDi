import os
from collections import OrderedDict
import time

import torch
import pytorch_lightning as pl

from utils import *
from arch.teacher_arch import TeacherNetwork
from losses import *

from kornia.losses import total_variation

class TeacherModel(pl.LightningModule):
    def __init__(self, save_path=None, lr=1e-4):
        super().__init__()
        self.model = TeacherNetwork()
        self.save_path = save_path
        
        self.lr = lr
        
        # Losses
        # Self supervised spatial exposure control loss
        self.cri_sec = SpaExpLoss(loss_weight=10)
        # Spatial consistency loss
        self.cri_sc = SpatialLoss(loss_weight=1)
        # Color consistency loss
        self.cri_cc = ColorLoss(loss_weight=5)
        # Illumination smoothness loss
        self.cri_is = IlluSmoothLoss(loss_weight=200)

    def training_step(self, batch, batch_idx):
        x, image_name = batch
        x_map = get_random_expmap(x)

        x_concat = torch.cat((x, x_map), dim=1)
        
        l_total = 0
        losses = OrderedDict()
        
        # Forward
        res, r = self.model(x_concat)
        res = (res + 1) / 2
        res_map = get_expmap(res, s=0.25)
        
        # Losses
        l_sec = torch.mean(self.cri_sec(res, x_map))
        l_total += l_sec
        losses["l_sec"] = l_sec
        
        # x = [0, 1], res = [-1, 1]
        l_sc = self.cri_sc(x, res)
        l_total += l_sc
        losses["l_sc"] = l_sc

        l_cc = self.cri_cc(res)
        l_total += l_cc
        losses["l_cc"] = l_cc

        l_is = self.cri_is(r)
        l_total += l_is
        losses["l_is"] = l_is
        
        losses["l_total"] = l_total
        
        self.log_dict(losses)
        return l_total


    def validation_step(self, batch, batch_idx):
        x, image_names = batch
        x_map = get_expmap(x)
        x_concat = torch.cat((x, x_map), dim=1)
        res, r = self.model(x_concat)
        res_map = get_expmap(res)

        return {"res": res, "res_map": res_map, "image_names": image_names}

    def validation_epoch_end(self, outputs):
        for output in outputs:
            for res, res_map, image_name in zip(output["res"], output["res_map"], output["image_names"]):
                # u_name = f'{self.current_epoch}/{image_name}_res_map.png'
                # u_path = os.path.join(self.save_path, u_name)
                # save_images(res_map, u_path)
                u_name = f'{self.current_epoch}/{image_name}_res.png'
                u_path = os.path.join(self.save_path, u_name)
                save_images(res, u_path, min_max=(-1, 1))
            

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
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
        res = self.model(x)
        u_name = f'{self.current_epoch}/{image_name}_inference.png'
        u_path = os.path.join(self.save_path, u_name)
        save_images(res, u_path)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer
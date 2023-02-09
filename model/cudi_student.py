import os
from collections import OrderedDict
import time

import torch
import pytorch_lightning as pl

from utils import *
from arch.student_arch import StudentNetwork
from losses import *


class StudentModel(pl.LightningModule):
    def __init__(self, save_path=None, lr=5e-4, teacher_weight=None):
        super().__init__()
        self.model = StudentNetwork()
        # Init model weight in normal distribution
        self.model.init_weight()
        self.save_path = save_path
        
        self.lr = lr

        # Losses
        self.cri_tea = StudentLoss(teacher_weight=teacher_weight)

    def training_step(self, batch, batch_idx):
        x, _, image_name = batch
        
        l_total = 0
        losses = OrderedDict()
        
        # Forward
        res = self.model(x)

        l_tea = self.cri_tea(x, res)
        l_total += l_tea
        losses["l_tea"] = l_tea

        
        self.log_dict(losses)
        return l_total


    def validation_step(self, batch, batch_idx):
        x, _, image_names = batch
        res = self.model(x)

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
        x = batch
        start = time.perf_counter()
        res = self.model(x[0])
        inf_time = time.perf_counter() - start
        
        return inf_time

    def test_epoch_end(self, outputs):
        avg = sum(outputs[1:]) / (len(outputs) - 1)
        print(f"Average inference time: {avg}")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, image_name = batch
        image_name = image_name[0].split('\\')[-1].split('.')[0]
        res = self.model(x)

        u_name = f'{self.current_epoch}/{image_name}_inference.png'
        u_path = os.path.join(self.save_path, u_name)
        save_images(res, u_path, min_max=(0, 1))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer
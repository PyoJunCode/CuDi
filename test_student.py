import os
import time
import argparse
import shutil

import torch
from torch import utils
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader.datamodule import PairedDataModule, CustomDataModule
from model.cudi_student import StudentModel


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("-e", type=str, default=None)
parser.add_argument("-t", action="store_true")
args = parser.parse_args()

date_name = time.strftime('%Y%m%d-%H%M')
if args.e:
    date_name += f"_{args.e}"
image_path = f"/mnt/nas/sjp/cudi-inference/cudi_{date_name}"
os.makedirs(image_path, exist_ok=True)


model = StudentModel.load_from_checkpoint(args.model)
model.save_path = image_path


config = {
    "base_dir": "/mnt/nas/sjp/data/dataset1",
    "dir_info": {
        "train_dir": ("train/hdr", "*01_0.1s_3200.jpg"),
        "valid_dir": ("test/data", "*01_0.1s_3200.jpg"),
        "test_dir": ("test", "*01_0.1s_3200.jpg"),
    },
    "log_dir": "./tb_logs",
    "train_batch": 1,
}

# Load PL data module.
dm = PairedDataModule(config["base_dir"], dir_info=config["dir_info"], batch_size=config["train_batch"])
dm.setup(stage="test")

# Default length of loader is 10
test_loader = dm.test_dataloader()

# Set experiment name
exp_name = f"{date_name}_cudi"
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    )
if args.t:
    trainer.test(model, test_loader)
else:
    trainer.predict(model, test_loader)
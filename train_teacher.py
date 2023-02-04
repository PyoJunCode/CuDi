import os
import time
import argparse
import shutil

import torch
from torch import utils
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader.datamodule import CustomDataModule
from model.cudi_teacher import TeacherModel


parser = argparse.ArgumentParser()
parser.add_argument("-e", type=str, default=None)
parser.add_argument("--gpus", type=int, default=1)
args = parser.parse_args()

date_name = time.strftime('%Y%m%d-%H%M')
if args.e:
    date_name += f"_{args.e}"
image_path = f"/mnt/nas/sjp/cudi-exp/cudi_{date_name}"
os.makedirs(image_path, exist_ok=True)

# Save model architecture
shutil.copyfile("./arch/teacher_arch.py", os.path.join(image_path, "model.py"))

model = TeacherModel(save_path=image_path)

config = {
    "base_dir": "/mnt/nas/sjp/data/dataset1",
    "dir_info": {
        "train_dir": ("train/hdr", "*01_0.1s_3200.jpg"),
        "valid_dir": ("test/data", "*01_0.1s_3200.jpg"),
        "test_dir": ("test/data", "*01_0.1s_3200.jpg"),
    },
    "log_dir": "./tb_logs",
    "train_batch": 4,
}

# Save config
with open(os.path.join(image_path, "config.py"), "w") as f:
    f.write(str(config))


# Load PL data module.
dm = CustomDataModule(config["base_dir"], dir_info=config["dir_info"], batch_size=config["train_batch"])
dm.setup(stage="fit")

# Default length of loader is 10
train_loader = dm.train_dataloader()
valid_loader = dm.valid_dataloader()

# Set experiment name
exp_name = f"{date_name}_cudi"
log_dir = os.path.join(config["log_dir"], exp_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logger = TensorBoardLogger(config["log_dir"], name=exp_name)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=args.gpus,
    max_epochs=200,
    strategy="ddp",
    #strategy="ddp",
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    logger=logger,
    num_sanity_val_steps=0
    )
trainer.fit(model, train_loader, valid_loader)
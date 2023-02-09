import os
import time
import argparse
import shutil

import torch
from torch import utils
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader.datamodule import PairedDataModule
from model.cudi_student import StudentModel


parser = argparse.ArgumentParser()
parser.add_argument("-e", type=str, default=None)
parser.add_argument("--gpus", type=int, default=1)
args = parser.parse_args()

date_name = time.strftime('%Y%m%d-%H%M')
if args.e:
    date_name += f"_{args.e}"
image_path = f"./cudi-exp/student_{date_name}"
os.makedirs(image_path, exist_ok=True)

# Save model architecture
shutil.copyfile("./arch/student_arch.py", os.path.join(image_path, "model.py"))

config = {
    "base_dir": "/home/junpyo/dataset/brycen",
    "dir_info": {
        "train_dir": ("train", "*.jpg"),
        "valid_dir": ("test", "*.jpg"),
        "test_dir": ("test", "*.jpg"),
    },
    "log_dir": "./tb_logs",
    "train_batch": 32,
    "teacher_weight": "/home/junpyo/cudi/weights/brycen_net_g_10500.pth"
}

model = StudentModel(save_path=image_path, teacher_weight=config["teacher_weight"])

# Save config
with open(os.path.join(image_path, "config.py"), "w") as f:
    f.write(str(config))


# Load PL data module.
dm = PairedDataModule(config["base_dir"], dir_info=config["dir_info"], batch_size=config["train_batch"])
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

latest_checkpoint = ModelCheckpoint(
    filename="ckpt-{epoch}-{step}",
    monitor="step",
    mode="max",
    every_n_train_steps=520,
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=args.gpus,
    max_epochs=10000,
    strategy="ddp",
    #strategy="ddp",
    check_val_every_n_epoch=100,
    log_every_n_steps=10,
    logger=logger,
    num_sanity_val_steps=0,
    )
trainer.fit(model, train_loader, valid_loader)
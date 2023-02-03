import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataloader.img_dataset import ImageDataset

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, base_dir: str=None, dir_info=None, batch_size=2, patch=None):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.dir_info = dir_info

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_data = ImageDataset(img_dir=self.base_dir, dir_info=self.dir_info["train_dir"], patch=True)
            self.valid_data = ImageDataset(img_dir=self.base_dir, dir_info=self.dir_info["valid_dir"], max_len=15)
                
        
        if stage in (None, "test"):
            if self.test_dir:
                self.test_data = ImageDataset(img_dir=self.base_dir, dir_info=self.dir_info["test_dir"])
            else:
                print("No test_dir passed.")

        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def valid_dataloader(self):
        return DataLoader(self.valid_data, batch_size=1, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=0)
        
import os
import glob
import numpy as np 
import torch
from torch.utils.data import Dataset




class ScanNetDataset(Dataset):

    def __init__(
        self,
        dataset_path="/dtu/blackhole/0e/169006/ScanNet",
        split="train",
        egocentric=True,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1
    ):
        self.dataset_root = dataset_path

        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.metadata_root = os.path.join(self.script_dir, "metadata")

        self.split = split

        self.egocentric = egocentric
        self.transform = transform

        # TODO: implement the rest


    def __get_data_list(self):

        if self.split == 

    def __len__(self):

        
        
    
    def __getitem__(self):
        
        
        
        
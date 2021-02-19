#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:25:46 2021

@author: https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
"""
import os
import subprocess
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from src.constants import DATADIR, TRAINDIR, VALDIR, TESTDIR


class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.data_dir = os.path.join(DATADIR, 'MedNIST')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.MedNIST_folders = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']


    def prepare_data(self):
        # download eg MedNIST
        os.chdir(DATADIR)
        # MedNIST:
        if not os.path.exists(self.data_dir):
            if not all(folder in os.listdir(data_dir) for folder in MedNIST_folders):
                !git clone https://github.com/apolanco3225/Medical-MNIST-Classification.git
        
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = DatasetFolder(self.data_dir, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)
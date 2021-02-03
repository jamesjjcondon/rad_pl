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
from torchvision.datasets import MNIST
from torchvision import transforms
from src.constants import DATADIR, TRAINDIR, VALDIR, TESTDIR


class YourModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


    def prepare_data(self):
        # download eg MedNIST
        os.chdir(DATADIR)
        # CBIS-DDSM:
        !curl -L -o nbia-data-retriever-3.6.deb 'https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_3.6/nbia-data-retriever-3.6.deb'
        !curl -L -o CBIS-DDSM.tcia 'https://wiki.cancerimagingarchive.net/download/attachments/22516629/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia?version=1&modificationDate=1534787024127&api=v2' 
        # MedNIST:
            
        #!curl -L -o MedNIST.tar.gz 'https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz'

        # NIH CXR: png
        # 'https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737/images_001.tar.gz'
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)
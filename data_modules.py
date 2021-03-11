#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:25:46 2021

@author: https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
"""
import os
import subprocess
import math
import cv2
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from constants import DATADIR #TRAINDIR, VALDIR, TESTDIR

from IPython import embed

class MedNIST(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data_dir = os.path.join(DATADIR, 'MedNIST')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.MedNIST_folders = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']

        # self.prepare_data()
        # self.setup()
        
    def _loader(self, x):
        x = cv2.imread(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.moveaxis(x, -1, 0).astype(np.float32)
        return x

    def prepare_data(self):
        # MedNIST:
        if 'Medical-MNIST-Classification' not in os.listdir(DATADIR):
            subprocess.call(['git', '-C', DATADIR, 'clone', 'https://github.com/apolanco3225/Medical-MNIST-Classification.git',])
        
        else:
            print('\nMedical-MNIST already in {}'.format(DATADIR))
            
    def setup(self, stage):
        
        # Assign train/val datasets for use in dataloaders
        mednist = DatasetFolder(
            root=DATADIR.joinpath('Medical-MNIST-Classification/resized'),
            loader=self._loader,
            extensions='.jpeg'
            )
        gen = torch.Generator()
        gen = gen.manual_seed(self.hparams.seed)
        
        train_n = math.ceil(mednist.__len__() * .7)
        val_n = test_n = math.floor(mednist.__len__() * .15)
        
        self.train_set, self.val_set, self.test_set = random_split(mednist, [train_n, val_n, test_n], generator=gen)


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size)
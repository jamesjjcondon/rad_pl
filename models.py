#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:22:52 2020

@author: James Condon
"""
import gc
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision import transforms, models

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.auroc import auroc

from IPython import embed


class ImageNet_Pretrained_Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=6)
        
    def forward(self, x):
        return self.model(x)
        
        
class YourModel(pl.LightningModule):
    """ 
    """
    def __init__(self, hparams):
        super(YourModel, self).__init__()
        """ 
        Initialise custom objects here
        """
        self.hparams = hparams
        self.model = ImageNet_Pretrained_Model()
        self.accuracy = pl.metrics.Accuracy()
        self.AUC = auroc
        
    def forward(self, x):
        logits = self.model.forward(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        train_loss = self.your_loss(out, y)
    
        self.log('train_loss', train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        val_loss = self.your_loss(out, y) # or self.your_custom_loss(out, y)
        
        self.log('val_loss', val_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        gc.collect() #pytorch/issues/40911
        return val_loss
    
    def your_loss(self, out, y):
        # Your loss function
        return F.cross_entropy(out, y)
    
    def configure_optimizers(self):
        optimiser = Adam(
                self.parameters(), lr=self.hparams.lr,
                betas=(0.9, 0.999), eps=1e-8, 
                weight_decay=self.hparams.weight_decay, #45e-06, 
                amsgrad=False
                )

        if self.hparams.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                    optimiser, 
                    step_size=2,
                    gamma=.2)
            
            return [optimiser], [scheduler]
        
        elif self.hparams.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimiser,
                base_lr=self.hparams.lr,
                max_lr=0.0001, # top of cycle
                step_size_up=3, 
                step_size_down=None, 
                mode='triangular', 
                gamma=0.75, #exp_range only top of cycle 3/4 every epoch
                scale_fn=None, 
                scale_mode='cycle',
                cycle_momentum=False, 
                base_momentum=0.8, 
                max_momentum=0.9, 
                last_epoch=-1
                )
            return [optimiser], [scheduler]
        
        elif self.hparams.lr_schedule == 'ROP':       
            scheduler = {
                            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    optimiser,
                                    mode='min',
                                    factor=0.3, 
                                    patience=2, 
                                    verbose=True,
                                    threshold=1e-04,
                                    threshold_mode='rel',
                                    cooldown=0, 
                                    min_lr=1e-12,
                                    eps=1e-08),
                            'monitor': 'train_loss_epoch'
                            }
            return [optimiser], [scheduler]
        
        else:
            return [optimiser]

     


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
from torchvision import transforms
import pytorch_lightning as pl
from src.data_loading.drjc_datasets import BSSA_exams
from torch.optim import Adam
import src.modeling.layers as layers
from src.constants import VIEWS, VIEWANGLES


class YourModel(pl.LightningModule):
    """ 
    """
    def __init__(self, hparams):
        super(YourModel, self).__init__()
        """ 
        Initialise custom objects here
        """

        self.accuracy = pl.metrics.Accuracy()
        
    def forward(self, x):
        h = self.all_views_gaussian_noise_layer(x)
        result = self.four_view_resnet(h)
        h = self.all_views_avg_pool(result)

        # Pool, flatten, and fully connected layers
        h_cc = torch.cat([h[VIEWS.L_CC], h[VIEWS.R_CC]], dim=1) # CCs concatenated """
        h_mlo = torch.cat([h[VIEWS.L_MLO], h[VIEWS.R_MLO]], dim=1) # MLOs concatenated """

        h_cc = F.relu(self.fc1_cc(h_cc))
        h_mlo = F.relu(self.fc1_mlo(h_mlo))
        
        h_cc = self.output_layer_cc(h_cc)
        h_mlo = self.output_layer_mlo(h_mlo)

        h = {
            VIEWANGLES.CC: h_cc,
            VIEWANGLES.MLO: h_mlo,
        }
        
        return h
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        train_loss = F.nll_loss(out, y)
        
        # score for this batch:
        batch_accuracy = self.compute_preds_acc_no_cpu(out, y, self.hparams.train_bsize)
        self.log('train_loss', train_loss, prog_bar=True, logger=True, on_step=True,
        on_epoch=True)
        self.log('train_acc', batch_accuracy, prog_bar=True, logger=True, on_step=True,
        on_epoch=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)

        val_loss = F.nll_loss(out, y) # or self.your_custom_loss(out, y)
        
        # score for this batch:
        batch_accuracy = self.compute_preds_acc_no_cpu(out, y)
        
        self.log('val_loss', val_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val_acc', batch_accuracy, prog_bar=True, logger=True, on_step=True,
        on_epoch=True)
        gc.collect() #pytorch/issues/40911
        return val_loss
    
    def your_custom_loss(self, out, y):
        # Your custom loss function
        loss = False
        return loss
    
    def your_accuracy(self, output, targets, batch_size, mode='view_split'):
        # for multiple views or other requirements depending on your data.
        logits, class_preds = torch.max(output, dim=-1)
        return self.accuracy(class_preds, targets)
    
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

    def train_dataloader(self):
        ds = dicom_dataset(
                exam_list_in=self.hparams.train_exam_list_fp,
                transform=self.train_trsfm,
                parameters=self.hparams,
                train_val_or_test='train'
                )

        dloader = DataLoader(
                dataset=ds,
                batch_size=self.hparams.train_bsize, #4, #self.hparams.batch_size,
                num_workers=self.hparams.num_workers, #6,
                pin_memory=True                
                )
        return dloader

    def val_dataloader(self):
        ds = dicom_dataset(
                exam_list_in=self.hparams.val_exam_list_fp,
                transform=self.train_trsfm,
                parameters=self.hparams,
                train_val_or_test='val'
                )
        
        dloader = DataLoader(
                dataset=ds,
                batch_size=self.hparams.val_bsize,
                num_workers=4, #self.hparams.num_workers, #0, # >0 blows out RAM,
                pin_memory=True, 
                )
        return dloader

     


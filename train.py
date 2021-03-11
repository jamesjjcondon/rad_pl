#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:14:57 2021

@author: drjc
"""

from datetime import datetime
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.parsing import AttributeDict

from models import YourModel
from data_modules import MedNIST
from constants import DATADIR

from IPython import embed
#%%
     
def train(hparams):
    
    early_stop_callback = EarlyStopping(
            monitor='val_loss_epoch',
            min_delta=1,
            patience=10,
            verbose=True,
            mode='min',
            strict=False
            )
        
    date_time = datetime.now().strftime("%Y-%m-%d")
    
    ckpt_callback = ModelCheckpoint(
            dirpath=None,
            monitor='val_loss_epoch',
            verbose=1,
            save_top_k=5,
            save_weights_only=True,
            mode='min',
            period=1,
            filename='{epoch}-{train_loss_epoch:.3f}-{val_loss_epoch:.3f}'
            )
     
    logger = TestTubeLogger(save_dir=hparams.log_path,
            name=date_time)
    
    
    trainer = Trainer(
        accelerator=hparams.accel,
        amp_backend='native',
        auto_lr_find=hparams.autolr,
        benchmark=True,
        callbacks=[ckpt_callback, early_stop_callback],
        check_val_every_n_epoch=hparams.check_val_n,
        fast_dev_run=False,
        gpus=hparams.gpus,
        logger=logger,
        max_epochs=hparams.max_epochs,
        overfit_batches=16,
        precision=hparams.precision,
        profiler=False,
        )
    
    data = MedNIST(hparams)

    # embed()
    
    model = YourModel(hparams)
     
    trainer.fit(model, data)
    
if __name__ == "__main__":
    hparams = AttributeDict(
        {
            'accel': None,
            'autolr': False,
            'batch_size': 2,
            'check_val_n': 1,
            'dev': False,
            'gpus': -1,
            'grad_cum': False,
            'log_path': DATADIR.joinpath('logs'),
            'lr': 0.0001,
            'lr_schedule': 'ROP',
            'max_epochs': 100,
            'num_nodes': 1,
            'num_workers': 0,
            'pl_ver': pl.__version__,
            'precision': 16,
            'seed': 22117,
            'weight_decay': 1e-07
            })
    
    train(hparams)
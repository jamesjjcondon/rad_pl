# rad_pl

A pytorch_lightning template for radiology convolutional neural newtorks. 

# Installation:

## Ubuntu

1. Setup a virtualenvironment first:
- use either virtualenv or conda
- activate it

For example:
`python -m venv my_environments/rad_pl`

`source my_environments/rad_pl/bin/activate` 


2. Download the repo:

In a terminal, 

2.1 cd into a folder where you want to create this template

2.2. type `git clone https://github.com/jamesjjcondon/rad_pl.git`

2.3. `cd rad_pl`

2.4. `pip install -r requirements.txt`

## Other OS
You on your own, sorry.
Learn Ubuntu : )

## Adjust a folders / directories
Open 

## Start training

- from the rad_pl directory type `python train.py`

## Test out hyperparameters and pytorch_lightning.Trainer flags
https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html 

## Use tensorboard to view learning curves:
`tensorboard --logdir '...<your-DATADIR->\logs'`


if you want help with a dicom pytorch dataset, pm me.

Have fun!


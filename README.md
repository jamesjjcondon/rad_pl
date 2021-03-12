# rad_pl

A pytorch_lightning project template for radiology convolutional neural newtorks.

#ArtificialIntelligence
#NeuralNetworks
#HealthAI
#Pytorch 

# Installation:

## Ubuntu

## 1. Setup a virtual environment first:
- use either virtualenv or conda
- activate it

For example:
`python -m venv my_environments/rad_pl`

`source my_environments/rad_pl/bin/activate` 

## 2. Download the repo:

In a terminal, 

 - cd into a folder where you want to create this template

 - type 
   `git clone https://github.com/jamesjjcondon/rad_pl.git`

 - `cd rad_pl`

 - `pip install -r requirements.txt`

## Other OS
You're on your own, sorry.
Learn Ubuntu : )

## 3. Adjust constants.py for your folders / directories
Open constants.py and add in your 'machine' / computer name and a directory for DATADIR, then save.

(This makes it easy to move machines and share with colleagues.) 

# Start training on MedNIST
- care of Arturo Polanco: https://github.com/apolanco3225/Medical-MNIST-Classification
- from the rad_pl directory type 

`python train.py`

## Test out hyperparameters and pytorch_lightning.Trainer flags
Check out the trainer section on pytorch_lightning docs page. 
Currently: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html 

## Use tensorboard to view learning curves:
`tensorboard --logdir '...<your-DATADIR->\logs'`

## Do your own inference and evaluation

## Try customising to a different dataset and with different file formats
- learn to use IPython.embed() to debug! It'll save you a LOT of time.

if you want help with a dicom pytorch dataset, pm me.

Have fun!

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

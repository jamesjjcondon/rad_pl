# rad_pl

A pytorch_lightning template for radiology convolutional neural newtorks.

#ArtificialIntelligence
#NeuralNetworks
#HealthAI 

# Installation:

## Ubuntu

## 1. Setup a virtualenvironment first:
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

## 4. Start training

- from the rad_pl directory type `python train.py`

## 5. Test out hyperparameters and pytorch_lightning.Trainer flags
Check out the trainer section on pytorch_lightning docs page. 
Currently: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html 

## 6. Use tensorboard to view learning curves:
`tensorboard --logdir '...<your-DATADIR->\logs'`

## 7.Do your own inference and evaluation

## 8. Try customising to a different dataset and with different file formats
- learn to use IPython.embed() to debug! It'll save you a LOT of time.

if you want help with a dicom pytorch dataset, pm me.

Have fun!


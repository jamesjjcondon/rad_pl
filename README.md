## rad_pl

A [pytorch_lightning](https://www.pytorchlightning.ai/) project template for radiology convolutional neural newtorks.

The focus here is to expedite you getting started on your own project. 

#ArtificialIntelligence
#NeuralNetworks
#CNNs
#HealthAI
#Pytorch
#PytorchLightning

# Dataset - MedNIST:

MedNIST is made available by Dr. Bradley J. Erickson M.D., Ph.D. (Department of Radiology, Mayo Clinic) under the [Creative Commons CC BY-SA 4.0 license](https://creativecommons.org/licenses/by/4.0/).

If you use the MedNIST dataset, please acknowledge the source.
It originates and sizes have been standardised from:
 - [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/), 
   - Clark K, Vendt B, Smith K, et al. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging. 2013; 26(6): 1045-1057. doi: 10.1007/s10278-013-9622-7. ([link](https://pubmed.ncbi.nlm.nih.gov/23884657/))
 - [RSNA Bone Age Challenge](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017) 
   - Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al. The RSNA Pediatric Bone Age Machine Learning Challenge. Radiology 2018; 290(2):498-503.
 - [NIH Chest X-ray dataset]('https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community')
   - Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017, http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
   
Please see the relevant terms of use:
 - [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/access-data/)
 - [RSNA Bone Age Challenge Terms of Use](https://www.rsna.org/-/media/Files/RSNA/Education/AI-resources-and-training/AI-image-challenge/RSNA-2017-AI-Challenge-Terms-of-Use-and-Attribution_Final.ashx?la=en&hash=F28B401E267D05658C85F5D207EC4F9AE9AE6FA9)

The scripts here, as is, are simply classifiying the images' modality into these classes:
- Abdomen CT
- Breast MRI
- Chest X-ray
- Chest CT
- Hand XR
- Head CT

This is a relatively easy task for a model to learn.

# Prerequisites:
- [git](https://git-scm.com/downloads)
- a [python](https://www.python.org/downloads/) installation
- ability to create virtual environments
- NVIDIA's [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) if you have and want to use a GPU/graphics card.

# Installation:

## 1. Setup a virtual environment first:
- use either virtualenv, venv or conda
- activate it
- make sure you've got pip

For example with venv:

`python -m venv my_environments/rad_pl_env`

`source my_environments/rad_pl_env/bin/activate` 

or conda:

`conda create --name rad_pl_env`

`conda activate rad_pl_env`
and you might need to `conda install pip`

## 2. Download the repository:

In a terminal, 

 - cd into a folder where you want to create this template

 - type 
   `git clone https://github.com/jamesjjcondon/rad_pl.git`

 - `cd rad_pl`

 - `pip install -r requirements.txt`


## 3. Adjust constants.py for your folders / directories
Open constants.py and add in your 'machine' / computer name and a directory for DATADIR, then save.

On windows you can use 'vim' from an anaconda prompt or notepad to view and edit. 
(This makes it easy to move machines and share with colleagues.) 

In a terminal, you could also type:
`python constants.py` to prompt an error message with your machine name (to add back into constants.py).

## Start training on MedNIST
- care of Arturo Polanco: https://github.com/apolanco3225/Medical-MNIST-Classification
- from the rad_pl directory type:

`python train.py`

## Test out hyperparameters and pytorch_lightning.Trainer flags
Check out the trainer section on pytorch_lightning docs page. 
Currently: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html 

## Use tensorboard to view learning curves:
`tensorboard --logdir '...<your-DATADIR->\logs'`

eg `tensorboard --logdir C:\Users\James\Documents\rad_pl\logs`
then open up a browser at htttp://localhost:6006 (or wherever tensorboard tells you to).

## Do your own inference and evaluation

## Try customising to a different dataset and with different file formats
- learn to use IPython.embed() to debug! It'll save you a LOT of time.

# To Do:
 - [ ] Add generic inference and possibly evaluation scripts
 - [ ] Add a dicom dataset loader and/or class
 - [ ] Add a dicom --> hdf5 preprocessing torch.utils.data.DataLoader
 - [ ] add a hdf5 dataset class
 - [ ] Test on Windows
 
Have fun!

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

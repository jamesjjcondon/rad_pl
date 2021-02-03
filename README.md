# rad_pl

A pytorch_lightning template for radiology convolutional neural newtorks. 

Our approach is to keep train, validation and test-sets patient-level and in completely separate directories to reduce risk of leakage.

At least, you'll need to customise:
- Your machine name and directories in src.constants.py
- The dicom_dataset for your data.


# Installation:


## Ubuntu

Setup a virtualenvironment first:
- use either virtualenv or conda
- activate it

In a terminal, 

1. type git clone 'this repor address'

2. `cd rad_pl`

3. `pip install -r requirements.txt`

## Other OS
You on your own, sorry.
Learn Ubuntu : )

Have fun!

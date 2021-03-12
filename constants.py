"""
Defines directory constants used in other scripts.
Here you can establish various different machines that you work on and choose the relevant folders to store data.
This habit makes it easy to move machines and share with colleagues.

For the current MedNIST example, all data is stored in DATADIR, but a better practices is to have completely separate folders for training, validation and testing, to avoid 'leakage'.
"""

from pathlib import Path
import platform

machine = platform.node()

if machine == '<name-of-your-computer>': # eg user@computer_name / also the 'machine' variable in this script
    DATADIR = Path('C:\\Users\James\data\')

elif machine == 'work-computer': # / <name-of-your-home-computer>
    DATADIR = Path('/data1')
    TRAINDIR = Path('/media/james/data/train')
    VALDIR = Path('/media/james/data/val')
    TESTDIR = Path('/media/james/data/test')
    
elif machine == "<name of new computer here>":
    CSVDIR = Path('Directory for .csv or similar with ground truth data')    
    DATADIR = Path('Directory for your train data')
    TRAINDIR = Path('Directory for your validation data')
    TESTDIR = Path('Directory for your test data')
    
else:
    raise ValueError("This computer ({}) has not been added to constants.py. \nPlease add it and the directories for your data".format(machine))
    
    

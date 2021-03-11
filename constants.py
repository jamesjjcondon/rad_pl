"""
Defines constants used in src.
"""

from pathlib import Path
import platform

machine = platform.node()

if machine == '<name-of-your-work-computer':
    TRAINDIR = '/data/james/NYU_retrain'
    VALDIR = '/nvme/james/val_ims_master'
    TESTDIR = '/nvme/james/test_ims_master'

elif machine == 'home': # / <name-of-your-home-computer>
    DATADIR = Path('/data1')
    #TRAINDIR = VALDIR = TESTDIR = '/media/james/drjc_ext_HD1/data/NYU_retrain'
    
elif machine == "<name of new computer here>":# eg user@computer_name / also the 'machine' variable in this script
    CSVDIR = 'Directory for .csv or similar with ground truth data'    
    DATADIR = 'Directory for your train data'
    TRAINDIR = 'Directory for your validation data'
    TESTDIR = 'Directory for your test data'
    
else:
    raise ValueError("This computer ({}) has not been added to src.constants.py. \nPlease add it and the directories for your data".format(machine))
    
    
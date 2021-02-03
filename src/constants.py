"""
Defines constants used in src.
"""
import platform
machine = platform.node()

if machine == 'lambda-quad': #AIML
    CSVDIR = '/home/mlim-user/Documents/james/tempdir/cleaning_temp' 
    TRAINDIR = '/data/james/NYU_retrain'
    VALDIR = '/nvme/james/val_ims_master'
    TESTDIR = '/nvme/james/test_ims_master'

elif machine == 'home':
    CSVDIR = '/media/1TB_HDD/samlim/temp'
    TRAINDIR = VALDIR = TESTDIR = '/media/james/drjc_ext_HD1/data/NYU_retrain'

elif machine == 'DL136541':
    CSVDIR = '/Data/james/tmp'
    DATADIR = TRAINDIR = VALDIR = TESTDIR = '/Data/james/'
    
elif machine == "<name of new computer here>":# eg user@computer_name
    CSVDIR = 'Directory for .csv or similar with ground truth data'    
    DATADIR = 'Directory for your train data'
    TRAINDIR = 'Directory for your validation data'
    TESTDIR = 'Directory for your test data'
    
else:
    raise ValueError("This computer ({}) has not been added to src.constants.py. \nPlease add it and the directories for your data".format(machine))
    
    
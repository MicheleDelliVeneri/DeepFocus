import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from natsort import natsorted
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str,
                    help='The directory containing the fits files;', default='')
args = parser.parse_args()
data_dir = args.data_dir

print('Splitting fits fits in bands subdirectories')

euclid_lsst_dir = os.path.join(data_dir, 'euclid+lsst')
galex_dir = os.path.join(data_dir, 'galex')
wise_dir = os.path.join(data_dir, 'wise')
other_dir = os.path.join(data_dir, 'otherbands')
targets_dir = os.path.join(data_dir, 'targets')

filelist = np.array(natsorted(list([file for file in os.listdir(data_dir) if '.fits' in file])))

if os.path.exists(euclid_lsst_dir) is False:
    os.mkdir(euclid_lsst_dir)
if os.path.exists(galex_dir) is False:
    os.mkdir(galex_dir)
if os.path.exists(wise_dir) is False:
    os.mkdir(wise_dir)
if os.path.exists(other_dir) is False:
    os.mkdir(other_dir)
if os.path.exists(targets_dir) is False:
    os.mkdir(targets_dir)

for file in tqdm(filelist, desc='Moving files into subdirectories', total=len(filelist)):
    if 'Euclid' in file:
        shutil.move(os.path.join(data_dir, file), os.path.join(euclid_lsst_dir, file))
    elif 'LSST' in file:
        shutil.move(os.path.join(data_dir, file), os.path.join(euclid_lsst_dir, file))
    elif 'GALEX' in file:
        shutil.move(os.path.join(data_dir, file), os.path.join(galex_dir, file))
    elif 'WISE' in file:
        shutil.move(os.path.join(data_dir, file), os.path.join(wise_dir, file))
    elif ('2MASS' in file) or ('Johnson' in file):
        shutil.move(os.path.join(data_dir, file), os.path.join(other_dir, file))
    elif ('stellarage' in file) or ('stellarmass' in file) or ('stellarmetallicity' in file) or ('dustmass' in file) or ('sfr' in file):
        shutil.move(os.path.join(data_dir, file), os.path.join(targets_dir, file))

print('Finished!')

import os 
import sys
import dl_utils as dl
import numpy as np

def TNG_load_data(data_path, catalogue_path, bands, targets):
    band_dir = os.listdir(data_path)
    catalogue = dl.load_catalogue(catalogue_path)
    idlist = catalogue['subhalo ID'].values
    euclid_lsst_dir = os.path.join(data_path, 'euclid+lsst')
    galex_dir = os.path.join(data_path, 'galex')
    wise_dir = os.path.join(data_path, 'wise')
    other_dir = os.path.join(data_path, 'otherbands')
    targets_dir = os.path.join(data_path, 'targets')
    euclid_files = [file for file in os.listdir(euclid_lsst_dir) if 'Euclid' in file]
    lsst_files = [file for file in os.listdir(euclid_lsst_dir) if 'LSST' in file]
    galex_files = [file for file in os.listdir(galex_dir) if 'GALEX' in file]
    wise_files = [file for file in os.listdir(wise_dir) if 'WISE' in file]
    two_mass_files = [file for file in os.listdir(other_dir) if '2MASS' in file]
    johnson_files = [file for file in os.listdir(other_dir) if 'Johnson' in file]
    band_info = {}
    for band in bands:
        if 'Euclid' in band:
            band_files = [file for file in euclid_files if band in file]
        if 'LSST' in band:
            band_files = [file for file in lsst_files if band in file]
        if 'GALEX' in band:
            band_files = [file for file in galex_files if band in file]
        if 'WISE' in band:
            band_files = [file for file in wise_files if band in file]
        if 'MASS' in band:
            band_files = [file for file in two_mass_files if band in file]
        if 'Johnson' in band:
            band_files = [file for file in johnson_files if band in file]
       
        band_info[band] = {}
        band_info[band]['files'] = band_files
        band_info[band]['ids'] = np.array([int("".join([t for t in tid.split('_')[0] if t.isdigit()])) for tid in band_files])
        band_info[band]['orientations'] = np.array([int(tid.split('_')[1][1]) for tid in band_files])
        print(len(band_files), len(band_info[band]['ids']), len(band_info[band]['orientations']))
        band_info[band]['len'] = len(band_files) // len(np.unique(band_info[band]['orientations']))
    
    target_info = {}
    targets = [file for file in os.listdir(targets_dir) if targets in file]
    target_info['targets'] = targets
    target_info['ids'] = np.array([int("".join([t for t in tid.split('_')[0] if t.isdigit()])) for tid in targets])
    target_info['orientations'] = np.array([int(tid.split('_')[1][1]) for tid in targets])
    target_info['len'] = len(targets) // len(np.unique(target_info['orientations']))
    
    print('You have selected the following bands:')
    for key, value in band_info.items():
        print(key, value['len'], len(np.unique(value['orientations'])), len(value['ids']))
    print('\n')
    print('and the following target:')
    print(target_info['len'], len(np.unique(target_info['orientations'])), len(target_info['ids']))
    print('Number of simulations in catalogue: ', len(idlist))
    catalogue = catalogue[catalogue['ID'].isin(target_info['ids'])]
    idlist = catalogue['ID'].values
    print('Number of simulations in catalogue after filtering: ', len(idlist))
    


#euclid_ids = np.array([int("".join([t for t in tid.split('_')[0] if (t.isdigit())])) for tid in euclid_filelist])
#print(euclid_ids)


catalogue_path = '/ibiscostorage/mdelliveneri/EUCLID/catalog/galaxyCatalogue.txt'
path = '/ibiscostorage/mdelliveneri/EUCLID/results'
bands = ['MASS_H', '2MASS_J', '2MASS_Ks', 'Euclid_H', 'Euclid_J', 'Euclid_VIS', 'Euclid_Y', 'GALEX_FUV', 'GALEX_NUV', 'Johnson_B', 'Johnson_I', 
        'Johnson_R', 'Johnson_U', 'Johnson_V', 'LSST_g', 'LSST_i', 'LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'WISE_W1', 'WISE_W2']
targets = 'dustmass'
TNG_load_data(path, catalogue_path, bands, targets)





#complete_ids = []
#for id in idlist:
#    idxs = np.where(image_ids == id)[0].astype(int)
#    if len(idxs) // 5 == 26:
#        for i in idxs:
#            complete_ids.append(i)
#complete_ids = np.array(complete_ids)
#unique_ids = np.unique(image_ids[complete_ids])
#catalogue = catalogue[catalogue['ID'].isin(unique_ids)]
#idlist = catalogue['ID'].values
#print(files[0])
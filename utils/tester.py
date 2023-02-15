import os 
import sys
import dl_utils as dl
import numpy as np
from sklearn.model_selection import train_test_split
import torchio as tio
from astropy.io import fits
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def load_fits(inFile):
    """
    This function loads a fits file and returns the data.
    """
    hdu_list = fits.open(inFile)
    data = hdu_list[0].data
    hdu_list.close()
    return data

def get_TNG_dataloaders(data_path, catalogue_path, bands, targets, train_size, training_transforms=None, validation_transforms=None, batch_size=1, num_workers=4):
    """
    This function returns the training and validation dataloaders for the TNG dataset. 
    INPUTS:
        data_path: path to the directory containing the TNG data (string)
        catalogue_path: path to the TNG catalogue (string)
        bands: list of bands to load, the order matters (list of strings)
        targets: list of targets to load, the order matters (list of strings)
        train_size: fraction of the data to use for training (float)
        training_transforms: transforms to apply to the training data (torchio.transforms)
        validation_transforms: transforms to apply to the validation data (torchio.transforms)
        batch_size: number of samples per batch (int)
        num_workers: number of workers to use for data loading (int)
    OUTPUTS:
        train_loader: training dataloader (torch.utils.data.DataLoader)
        val_loader: validation dataloader (torch.utils.data.DataLoader)
        test_loader: test dataloader (torch.utils.data.DataLoader)

    """
    
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
            band_files = [os.path.join(euclid_lsst_dir, file) for file in euclid_files if band in file]
        if 'LSST' in band:
            band_files = [os.path.join(euclid_lsst_dir, file) for file in lsst_files if band in file]
        if 'GALEX' in band:
            band_files = [os.path.join(galex_dir, file) for file in galex_files if band in file]
        if 'WISE' in band:
            band_files = [os.path.join(wise_dir, file) for file in wise_files if band in file]
        if 'MASS' in band:
            band_files = [os.path.join(other_dir, file) for file in two_mass_files if band in file]
        if 'Johnson' in band:
            band_files = [os.path.join(other_dir, file) for file in johnson_files if band in file]
       
        band_info[band] = {}
        band_info[band]['files'] = band_files
        band_info[band]['ids'] = np.array([int("".join([t for t in tid.split('_')[0].split('/')[-1] if t.isdigit()])) for tid in band_files])
        band_info[band]['orientations'] = np.array([int(tid.split('_')[1][1]) for tid in band_files])
        band_info[band]['len'] = len(band_files) // len(np.unique(band_info[band]['orientations']))
    
    target_info = {}
    targets_files = [os.path.join(targets_dir, file) for file in os.listdir(targets_dir) if targets in file]
    target_info['targets'] = np.array(targets_files)
    target_info['ids'] = np.array([int("".join([t for t in tid.split('_')[0].split('/')[-1] if t.isdigit()])) for tid in targets_files])
    target_info['orientations'] = np.array([int(tid.split('_')[1][1]) for tid in targets_files])
    target_info['len'] = len(targets_files) // len(np.unique(target_info['orientations']))
    
    print('You have selected the following bands:')
    for key, value in band_info.items():
        print(key, value['len'], len(np.unique(value['orientations'])), len(value['ids']))
    print('\n')
    print('and the following target:')
    print(targets, target_info['len'], len(np.unique(target_info['orientations'])), len(target_info['ids']))
    print('Number of simulations in catalogue: ', len(idlist))
    catalogue = catalogue[catalogue['subhalo ID'].isin(target_info['ids'])]
    idlist = catalogue['subhalo ID'].values
    print('Number of simulations in catalogue after filtering based on targets: ', len(idlist))
    print('Removing simualtions with no target counterpart:')
    for key, value in band_info.items():
        temp_files = band_info[key]['files']
        temp_ids = band_info[key]['ids']
        temp_orientations = band_info[key]['orientations']
        band_info[key]['files'] = np.array([temp_files[i] for i in range(len(temp_files)) if band_info[key]['ids'][i] in idlist])
        band_info[key]['orientations'] = np.array([temp_orientations[i] for i in range(len(temp_orientations)) if band_info[key]['ids'][i] in idlist])
        band_info[key]['ids'] = np.array([temp_ids[i] for i in range(len(temp_ids)) if band_info[key]['ids'][i] in idlist])
        band_info[key]['len'] = len(band_info[key]['files']) // len(np.unique(band_info[key]['orientations']))
    
    for key, value in band_info.items():
        print(key, value['len'], len(np.unique(value['orientations'])), len(value['ids']))
    print('\n')

    inputs_catalogue = {}
    targets_catalogue = {}
    for id in idlist:
        inputs_catalogue[id] = {}
        targets_catalogue[id] = {}
        for band in bands:
            inputs_catalogue[id][band] = {}
            idx = np.where(band_info[band]['ids'] == id)[0]
            files = band_info[band]['files'][idx]
            orientations = band_info[band]['orientations'][idx]
            inputs_catalogue[id][band]['images'] = files
            inputs_catalogue[id][band]['orientations'] = orientations
        idx = np.where(target_info['ids'] == id)[0]
        targets_catalogue[id][targets] = {}
        targets_catalogue[id][targets]['images'] = target_info['targets'][idx]
        targets_catalogue[id][targets]['orientations'] = target_info['orientations'][idx]

    train_ids, test_ids = train_test_split(idlist, test_size=1 - train_size, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=((1 - train_size) * len(idlist)) / len(train_ids), random_state=42)

    def TNGReader(path):
        data = load_fits(path)
        data = np.nan_to_num(np.array(data)).astype(np.float32)
        affine = np.eye(4)
        return data, affine
    
    def get_path_from_dictionary(input_catalogue, id, orientation, bands):
        catalogue = input_catalogue[id]
        paths = []
        for band in bands:
            idx = np.where(catalogue[band]['orientations'] == orientation)
            paths.append(catalogue[band]['images'][idx][0])
        return paths
    
    def create_subjects_dataset(inputs_catalogue, targets_catalogue, ids, target_info, targets, bands, transforms, mode='Training'):
        n_orientations = len(np.unique(target_info['orientations']))
        if isinstance(targets, list):
            pass
        else:
            targets = [targets]

        samples = []
        for id in tqdm(ids, desc='Storing {} samples into the dataloader'.format(mode), total=len(ids)):
            orientations = np.arange(1, n_orientations + 1)
            np.random.shuffle(orientations)
            for orientation in orientations:
                paths = get_path_from_dictionary(inputs_catalogue, id, orientation, bands)
                target_paths = get_path_from_dictionary(targets_catalogue, id, orientation, targets)
                sample = tio.Subject(
                    input=tio.ScalarImage(paths, reader=TNGReader),
                    target=tio.ScalarImage(target_paths, reader=TNGReader),
                    id=id,
                    orientation=orientation
                )
                samples.append(sample)
        dataset = tio.SubjectsDataset(samples, transform=transforms)
        return dataset

    training_dataset = create_subjects_dataset(inputs_catalogue, targets_catalogue, train_ids, target_info, targets, bands, training_transforms, mode='Training')
    validation_dataset = create_subjects_dataset(inputs_catalogue, targets_catalogue, val_ids, target_info, targets, bands, validation_transforms, mode='Validation')
    test_dataset = create_subjects_dataset(inputs_catalogue, targets_catalogue, test_ids, target_info, targets, bands, validation_transforms, mode='Test')
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return training_dataloader, validation_dataloader, test_dataloader

catalogue_path = '/ibiscostorage/mdelliveneri/EUCLID/version2/catalog/galaxyCatalogue.txt'
path = '/ibiscostorage/mdelliveneri/EUCLID/version2/results'
all_bands = ['MASS_H', '2MASS_J', '2MASS_Ks', 'Euclid_H', 'Euclid_J', 'Euclid_VIS', 'Euclid_Y', 'GALEX_FUV', 'GALEX_NUV', 'Johnson_B', 'Johnson_I', 
        'Johnson_R', 'Johnson_U', 'Johnson_V', 'LSST_g', 'LSST_i', 'LSST_r', 'LSST_u', 'LSST_y', 'LSST_z', 'WISE_W1', 'WISE_W2']

bands = ['GALEX_FUV', 'GALEX_NUV', 'LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'Euclid_Y','Euclid_J', 'Euclid_H']
batch_size = 32
output_path = '/ibiscostorage/mdelliveneri/EUCLID/version2/plots/examples'
if os.path.exists(output_path) == False:
    os.mkdir(output_path)

target_name = 'dustmass'
train_dataloader, validation_dataloader, test_dataloader = get_TNG_dataloaders(path, catalogue_path, bands, target_name, train_size=0.8, batch_size=batch_size)
batch = next(iter(train_dataloader))
inputs = torch.permute(batch['input'][tio.DATA], (0, 4, 2, 3, 1)).numpy()
targets = torch.permute(batch['target'][tio.DATA], (0, 4, 2, 3, 1)).numpy()
ids = batch['id']
orientations = batch['orientation']

for i_batch, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Plotting Samples'):
    inputs = torch.permute(batch['input'][tio.DATA], (0, 4, 2, 3, 1)).numpy()
    targets = torch.permute(batch['target'][tio.DATA], (0, 4, 2, 3, 1)).numpy()
    ids = batch['id']
    orientations = batch['orientation']
    for i in tqdm(range(batch_size), total=batch_size, desc='Plotting batch {}'.format(i_batch), leave=False):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        input_ = inputs[i][0]
        target = targets[i][0]
        integrated = np.sum(input_, axis=2)
        spectrum = np.sum(input_, axis=(0, 1))
        integrated = np.log10(integrated + np.min(integrated) + 1e-10)
        target = np.sum(target, axis=2)
        target = np.log10(target + np.min(target) + 1e-10)
        im0=ax[0].imshow(integrated, origin='lower', cmap='magma', label='Input')
        plt.colorbar(im0, ax=ax[0])
        im1=ax[1].imshow(target, origin='lower', cmap='magma', label='Target')
        plt.colorbar(im1, ax=ax[1])
        ax[0].set_title('Input Integrated Bands ID: {}, Orientation: {}'.format(ids[i], orientations[i]))
        ax[1].set_title('Target {}'.format(target_name))
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        plt.savefig(os.path.join(output_path, 'Input_Output_Comparison_{}_{}_{}.png'.format(i, ids[i], orientations[i])))
        plt.close()
        plt.figure(figsize=(10, 5))
        plt.plot(spectrum)
        plt.title('Input ID: {}, Orientation: {}'.format(ids[i], orientations[i]))
        plt.savefig(os.path.join(output_path, 'Input_Spectrum_{}_{}_{}.png'.format(i, ids[i], orientations[i])))
        plt.close()

        fig, axs = plt.subplots(nrows=2, ncols=len(bands) // 2, figsize=(5 * len(bands) // 2, 10))
        for k in range(2):
            for j in range(len(bands)  // 2):
                im = axs[k, j].imshow(np.log(input_[:, :, j] + np.min(input_[:, :, j]) + 1e-10), origin='lower', cmap='magma')
                axs[k, j].set_title('{}'.format(bands[j]))
                axs[k, j].set_xlabel('x')
                axs[k, j].set_ylabel('y')
                plt.colorbar(im, ax=axs[k, j])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'Input_Bands_{}_{}_{}.png'.format(i, ids[i], orientations[i])))
        plt.close()

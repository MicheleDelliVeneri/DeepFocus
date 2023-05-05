from astropy.io.fits import getheader
from astropy.io import fits
import numpy as np
import torch
from tqdm import tqdm
from itertools import starmap
from abc import ABC, abstractmethod
import pandas as pd
import pickle
from spectral_cube import SpectralCube
from astropy.wcs import WCS
import os
import utils.model_utils as mu
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class CubeCache:
    def __init__(self, fits_file: str, gradual_loading: bool = True):
        self.fits_file = fits_file
        self.hi_cube_tensor = None
        self.gradual_loading = gradual_loading

    def set_gradual_loading(self, value: bool):
        self.gradual_loading = value

    def comp_statistics(self, channels, percentiles=None):
        if percentiles is None:
            percentiles = [.1, 99.9]

        scale = list()
        mean = list()
        std = list()
        print('Computing Cube Statistics...')
        for channel in tqdm(channels, total=len(channels)):
            hi_data_fits = fits.getdata(self.fits_file, ignore_blank=True)
            comp_percentiles = np.percentile(hi_data_fits[channel], percentiles)
            clipped = np.clip(hi_data_fits[channel], *comp_percentiles)
            clipped = (clipped - comp_percentiles[0]) / (comp_percentiles[1] - comp_percentiles[0])

            scale.append(torch.tensor(comp_percentiles, dtype=torch.float32))
            mean.append(torch.tensor(clipped.mean(), dtype=torch.float32))
            std.append(torch.tensor(clipped.std(), dtype=torch.float32))

        return scale, mean, std

    def get_hi_data(self):
        return self.hi_cube_tensor

    def cache_data(self, slices):
        if self.gradual_loading:
            f0, f1 = slices[-1].start, slices[-1].stop
            self.hi_cube_tensor = torch.empty(tuple(map(lambda x: x.stop - x.start, slices)))

            for i, f in enumerate(tqdm(range(f0, f1), desc='Loading input data')):
                hi_data_fits = fits.getdata(self.fits_file, ignore_blank=True)
        
                self.hi_cube_tensor[:, :, i] = torch.tensor(hi_data_fits[f].astype(np.float32),
                                                            dtype=torch.float32).T[slices[:2]]
        else:
            hi_data_fits = fits.getdata(self.fits_file, ignore_blank=True)
            f0, f1 = slices[-1].start, slices[-1].stop
            self.hi_cube_tensor = torch.tensor(hi_data_fits[f0:f1].astype(np.float32), dtype=torch.float32).T[
                slices[:2]]

def coordinates_expand(dim, upper_left, cube_shape, padding):
    coord_start = ((dim - 2 * padding) * upper_left).astype(np.int32)
    coord_end = (coord_start + dim).astype(np.int32)

    keep = True

    for i, (e, s, d) in enumerate(zip(coord_end, cube_shape, dim)):
        if e < s < e + d:
            coord_end[i] = s
        elif e > s:
            keep = False

    return coord_start, coord_end, keep

def coordinates(dim, upper_left, cube_shape, padding):
    coord_start = ((dim - 2 * padding) * upper_left).astype(np.int32)
    coord_end = (coord_start + dim).astype(np.int32)

    ext_padding = np.zeros(3, dtype=np.int32)

    for i, (e, s) in enumerate(zip(coord_end, cube_shape)):
        ext_padding[i] = e - min(e, s)
        coord_end[i] = min(e, s)

    return coord_start, coord_end, ext_padding

def _partition_indexing(cube_shape, dim, padding, max_batch_size=None):
    if np.any(dim < 2 * padding):
        raise ValueError('Padding has to be less than half dimension')
    effective_shape = tuple(starmap(lambda s, p: s - 2 * p, zip(dim, padding)))
    patches_each_dim = tuple(starmap(lambda e, c, p: np.ceil((c - 2 * p) / e),
                                     zip(effective_shape, cube_shape, padding)))

    meshes = np.meshgrid(*map(np.arange, patches_each_dim))
    #meshes = [meshes[0], meshes[2], meshes[1]]
    upper_lefts = np.stack(list(map(np.ravel, meshes)))
    n_evaluations = upper_lefts.shape[1]
    if max_batch_size != None:
        batch_size = min(max_batch_size, n_evaluations)
    else:
        batch_size = n_evaluations

    n_index = int(np.ceil(float(n_evaluations) / batch_size))
    indexes_partition = np.array_split(np.arange(n_evaluations), n_index)
    return upper_lefts, indexes_partition

def partition_expanding(cube_shape, dim, padding):
    upper_lefts, indexes_partition = _partition_indexing(cube_shape, dim, padding)

    exp_slices_partition = list()
    for indexes in indexes_partition:
        exp_slices = list()
        for i, index in enumerate(indexes):
            c_start, c_end, keep = coordinates_expand(dim, upper_lefts[:, index], cube_shape, padding)
            if keep:
                exp_slices.append(list(starmap(lambda s, e: slice(s, e), zip(c_start, c_end))))
        if len(exp_slices) > 0:
            exp_slices_partition.append(exp_slices)
    return exp_slices_partition

def partition_overlap(cube_shape, dim, padding, max_batch_size=None):
    upper_lefts, indexes_partition = _partition_indexing(cube_shape, dim, padding, max_batch_size)

    overlap_slices_partition = list()
    overlaps_partition = list()
    for indexes in indexes_partition:
        overlap_slices = list()
        overlaps = list()
        for i, index in enumerate(indexes):
            c_start, c_end, overlap = coordinates(dim, upper_lefts[:, index], cube_shape, padding)
            overlaps.append(overlap)
            overlap_slices.append(list(starmap(lambda s, e, o: slice(s - o, e), zip(c_start, c_end, overlap))))
        overlap_slices_partition.append(overlap_slices)
        overlaps_partition.append(overlaps)
    return overlap_slices_partition, overlaps_partition

def _slice_add(slice_1, slice_2):
    return [slice(s1.start + s2.start + s2.stop, s1.stop + 2 * s2.stop) for s1, s2 in zip(slice_1, slice_2)]

def cube_evaluation(cube, dim, padding, position, overlap_slices, overlaps, model):
    model_input = torch.empty(len(overlap_slices), 1, *dim)
    frequency_channels = torch.empty((len(overlap_slices), 2))

    padding_slices = list()

    for i, ovs in tqdm(enumerate(overlap_slices), total=len(overlap_slices)):
        model_input[i, 0] = cube[ovs]
        frequency_channels[i, :] = torch.tensor([position[0, -1] + ovs[-1].start,
                                                 position[0, -1] + ovs[-1].stop])
        padd_slices = [slice(int(p + o), int(- p)) for o, p in zip(overlaps[i], padding)]
        padding_slices.append(padd_slices)

    
    model.eval()
    with torch.no_grad():
        model_out = model(model_input, frequency_channels)
    out = torch.empty(len(overlap_slices), 1, *dim)
    out[:, :, :, :, :] = model_out.detach().clone()
    del model_out
    torch.cuda.empty_cache()

    outputs = [m[0][p] for m, p in zip(out, padding_slices)]
    efficient_slices = [_slice_add(s, p) for s, p in zip(overlap_slices, padding_slices)]
    return outputs, efficient_slices

def connect_outputs(cube, outputs, efficient_slices, padding):
    eval_shape = tuple(starmap(lambda s, p: int(s - 2 * p), zip(cube.shape, padding)))
    eval_cube = torch.empty(eval_shape)
    for out, sli in zip(outputs, efficient_slices):
        eval_cube[sli] = out
    return eval_cube

def prepare_dir(directory):
    splitted = directory[1:].split('/')
    for i in range(1, len(splitted) + 1):
        d = '/' + '/'.join(splitted[:i])
        if not os.path.exists(d):
            os.mkdir(d)

class ModelTraverser(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def traverse(self, *args, **kwargs):
        pass

class SimpleModelTraverser(ModelTraverser):

    def __init__(self, model, cube: torch.tensor, padding, input_dim):
        super().__init__(model)
        self.cube = cube
        self.padding = padding
        self.input_dim = input_dim

    def traverse(self, index) -> dict:
        pass

class EvaluationTraverser(ModelTraverser):

    def __init__(self, model, fits_path, model_input_dim, dl_padding,
                 desired_dim, max_batch_size,
                 multi_node: bool = False,
                 multi_gpu : bool = False,
                 n_parallel: int = 1, 
                 i_job: int = 0, j_loop: int = 0, df_name: str = None):
        super().__init__(model)

        # Understand where we are running
        self.multi_node = multi_node
        self.multi_gpu = multi_gpu
        if self.multi_node == True:
            self.global_rank = int(os.environ['RANK'])
            # Force multi gpu to True
            self.multi_gpu == True
        elif self.multi_gpu == True:
             self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.local_rank = device
        self.model_input_dim = model_input_dim
        self.header = fits.getheader(fits_path, ignore_blank=True)
        self.cube_shape = np.array(list(map(lambda x: self.header[x], ['NAXIS1', 'NAXIS2', 'NAXIS3'])))
        print('Detected cube shape: ', self.cube_shape)
        self.desired_dim = np.minimum(desired_dim, self.cube_shape)
        self.padding = dl_padding 
        self.n_parallel = n_parallel
        self.i_job = i_job
        self.j_loop = j_loop
        self.max_batch_size = max_batch_size
        self.data_cache = CubeCache(fits_path)
        slices_partition = partition_expanding(self.cube_shape, desired_dim + 2 * self.padding, self.padding)
        self.slices_partition = np.array_split(slices_partition[0], n_parallel)[i_job]
        if df_name is None:
            df_name = ''
        self.df_name = df_name + '_n_parallel' + str(n_parallel) + '_i_job' + str(i_job) + '.txt'

    def __len__(self):
        return len(self.slices_partition)

    def traverse(self, save_output=False, save_input=False, remove_cols=True, output_path=None) -> pd.DataFrame:
        if self.j_loop > 0:
            df = pd.read_csv(self.df_name)
        else:
            df = pd.DataFrame()

        for j, slices in tqdm(enumerate(self.slices_partition), total=len(self.slices_partition), desc='Looping'):
            #print('Loop {} of {}'.format(str(j), str(len(self.slices_partition))))
            if j >= self.j_loop:
                self.data_cache.cache_data(slices)
                hi_cube_tensor = self.data_cache.get_hi_data()
                position = np.array([[s.start, s.stop] for s in slices]).T
                overlap_slices_partition, overlaps_partition = partition_overlap(position[1] - position[0],
                                                                                 self.model_input_dim,
                                                                                 self.padding, self.max_batch_size)
                outputs = list()
                efficient_slices = list()
                for overlap_slices, overlaps in tqdm(zip(overlap_slices_partition, overlaps_partition),
                                                     total=len(overlap_slices_partition), desc='Propagating model'):
                    try:
                        o, e = cube_evaluation(hi_cube_tensor, self.model_input_dim, self.padding, position,
                                               overlap_slices, overlaps, self.model)
                    except:
                        pickle.dump({'j_loop': j, 'n_parallel': self.n_parallel, 'i_job': self.i_job},
                                    open("j_loop.p", "wb"))
                        raise ValueError('Memory Issue')
                    outputs += o
                    efficient_slices += e    
                mask = connect_outputs(hi_cube_tensor, outputs, efficient_slices, self.padding)
                del outputs
                inner_slices = [slice(p, -p) for p in self.padding]
                hi_cube_tensor = hi_cube_tensor[inner_slices]

                partition_position = torch.tensor([[s.start + p for s, p in zip(slices, self.padding)],
                                                   [s.stop - p for s, p in zip(slices, self.padding)]])
                inner_slices.reverse()
                wcs = WCS(self.header)[inner_slices]
                model_out_fits = SpectralCube(mask.T.cpu().numpy(), wcs, header=self.header)
                prepare_dir(f'{output_path}/model_out')
                if os.path.isfile(f'{output_path}/model_out/{j}.fits'):
                    os.remove(f'{output_path}/model_out/{j}.fits')
                model_out_fits.write(f'{output_path}/model_out/{j}.fits', format='fits')
                del model_out_fits
                prepare_dir(f'{output_path}/partition_position')
                if os.path.isfile(f'{output_path}/partition_position/{j}.pb'):
                    os.remove(f'{output_path}/partition_position/{j}.pb')
                torch.save(partition_position, f'{output_path}/partition_position/{j}.pb')

                prepare_dir(f'{output_path}/slices')
                if os.path.isfile(f'{output_path}/slices/{j}.pb'):
                    os.remove(f'{output_path}/slices/{j}.pb')
                pickle.dump(slices, open(f'{output_path}/slices/{j}.pb', 'wb'))
                continue

def get_statistics(fits_path):
    header = fits.getheader(fits_path, ignore_blank=True)
    cc = CubeCache(fits_path)
    scale, mean, std = cc.comp_statistics(np.arange(header['NAXIS3']))
    return scale, mean, std

def get_base_segmenter(fits_path, input_shape, multi_gpu=False, multi_node=False):
    scale, mean, std = get_statistics(fits_path)
    if multi_node is True:
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        multi_gpu = True
    elif multi_gpu is True and multi_node is False:
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank  = device
        global_rank = None
    model = mu.DeepFocus(scale=scale, mean=mean, std=std, input_shape=input_shape, 
                         global_rank=global_rank, local_rank=local_rank, 
                         multi_gpu=multi_gpu)
    model = model.to(local_rank)
    if multi_gpu is True:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    return model


from astropy.io.fits import getheader
from astropy.io import fits
import numpy as np
import torch
from tqdm import tqdm
from itertools import starmap



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

        for channel in channels:
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
                self.hi_cube_tensor[:, :, i] = torch.tensor(hi_data_fits[0, f].astype(np.float32),
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
    if max_batch_size is not None:
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

n_parallel = 1
i_job = 0
max_batch_size = 32
model_input_dim = np.array([128, 128, 256])
cnn_padding = np.array([8, 8, 8])
desired_dim = (model_input_dim - 2 * cnn_padding)
fits_file = '/ibiscostorage/mdelliveneri/ALMA/SpeedComparison/member.uid___A001_X2c9_X1e.ari_l.NGC4395_sci.spw3_230289MHz.12m.cube.I.pbcor.fits'
header = fits.getheader(fits_file)
cube_shape = np.array(list(map(lambda x: header[x], ['NAXIS1', 'NAXIS2', 'NAXIS3'])))
desired_dim = np.minimum(desired_dim, cube_shape)
print('Cube Shape: ', cube_shape)
print('Desired Dim: ', desired_dim)
data_cache = CubeCache(fits_file, gradual_loading=True)
slices_partition = partition_expanding(cube_shape, desired_dim + 2 * cnn_padding, cnn_padding)
slices_partition = np.array_split(slices_partition[0], n_parallel)[i_job]

for j, slices in enumerate(slices_partition):
        print('Loop {} of {}'.format(str(j), str(len(slices_partition))))
        print(slices)
        data_cache.cache_data(slices)
        hi_cube_tensor = data_cache.get_hi_data()
        position = np.array([[s.start, s.stop] for s in slices]).T
        overlap_slices_partition, overlaps_partition = partition_overlap(position[1] - position[0],
                                                                            model_input_dim,
                                                                            cnn_padding, 
                                                                            max_batch_size)
        for overlap_slices, overlaps in tqdm(zip(overlap_slices_partition, overlaps_partition),
                                                     total=len(overlap_slices_partition), desc='Propagating model'):
            model_input = torch.empty(len(overlap_slices), 1, *model_input_dim)
            frequency_channels = torch.empty((len(overlap_slices), 2))
            padding_slices = list()
            print(model_input.shape, frequency_channels.shape)
            #for i, ovs in enumerate(overlap_slices):
                #model_input[i, 0] = hi_cube_tensor[ovs]
                #frequency_channels[i, :] = torch.tensor([position[0, -1] + ovs[-1].start,
                #                                 position[0, -1] + ovs[-1].stop])
                #padd_slices = [slice(int(p + o), int(- p)) for o, p in zip(overlaps[i], cnn_padding)]
                #padding_slices.append(padd_slices)
            #print(model_input.shape, frequency_channels.shape)
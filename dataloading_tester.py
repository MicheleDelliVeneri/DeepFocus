import utils.dataloading as dlt
import utils.model_utils as mu
from astropy.io.fits import getheader
from astropy.io import fits
import numpy as np
import torch
from tqdm import tqdm
from itertools import starmap

path = '/media/storage/big_cube/dirty_cube_0.fits'
n_parallel = 1
i_job = 0
max_batch_size = 32
header = fits.getheader(path)
model_input_dim = np.array([128, 128, 128])
cnn_padding = np.array([8, 8, 8])
desired_dim = (model_input_dim - 2 * cnn_padding)
multi_gpu = False
multi_node = False
segmenter = dlt.get_base_segmenter(path, (128, 128, 128), multi_gpu, multi_node)
device = torch.device('cuda')
evaluator = dlt.EvaluationTraverser(segmenter, path, model_input_dim, cnn_padding,
                                     desired_dim, max_batch_size, n_parallel=1, i_job=0)
df = evaluator.traverse(output_path='/media/storage/big_cube/processed.fits')
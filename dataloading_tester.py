import utils.dataloading as dlt
import utils.model_utils as mu
import utils.dl_utils as dl
from astropy.io.fits import getheader
from astropy.io import fits
import numpy as np
import torch
from tqdm import tqdm
from itertools import starmap

if __name__ == "__main__":
    dl.ddp_setup()
    path = '/media/storage/big_cube/dirty_cube_0.fits'
    n_parallel = 20
    i_job = 10
    max_batch_size = None
    header = fits.getheader(path)
    model_input_dim = np.array([256, 256, 512])
    cnn_padding = np.array([8, 8, 8])
    desired_dim = (model_input_dim - 2 * cnn_padding)
    multi_gpu = True
    multi_node = False
    segmenter = dlt.get_base_segmenter(path, model_input_dim, multi_gpu, multi_node)
    evaluator = dlt.EvaluationTraverser(segmenter, path, model_input_dim, cnn_padding,
                                     desired_dim, max_batch_size, n_parallel=n_parallel, i_job=i_job)
    df = evaluator.traverse(output_path='/media/storage/big_cube/processed.fits')
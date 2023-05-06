import pprint
import wandb
import utils.dl_utils as dl
import torchio as tio
import torch.multiprocessing as mp


config = dict(
    project = 'DeepFocus',
    entity = 'almadl',
    group = 'multigpu',
    name = 'multigpou_test_02',
    dataset = 'ALMA',
    inference_path = '/ibiscostorage/mdelliveneri/load_test_cube.fits',
    output_path = '/ibiscostorage/mdelliveneri/speed_test',
    plot_path = '/ibiscostorage/mdelliveneri/speed_test/plots/',
    batch_size = 8,
    num_workers = 4, 
    num_gpus = 2,
    dmode = 'deconvolver',
    block_sizes = [64, 128, 256, 512],
    oblock_sizes = [64, 32, 1],
    kernel_sizes = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
    depths = [2, 2, 2, 2],
    output_kernel_sizes = [3, 3, 3],
    hidden_size = 1024,
    encoder_activation = 'leaky_relu',
    decoder_activation = 'leaky_relu',
    block = 'basic', # basic, bottleneck
    skip_connections = False,
    in_channels = 1,
    out_channels = 1,
    dropout_rate = 0.0,
    input_shape = (256, 256, 128),
    patch_overlap = (8, 8, 8),
    final_activation = 'sigmoid',
    multi_gpu = True,
    multi_node = False, 
    log_rate = 1,
    save_frequency = 1,
    
    num_nodes = 1,
    debug = False,
    identity = False
)

if __name__ == "__main__":
    dl.inference_multigpu(config)
import pprint
import wandb
import utils.dl_utils as dl
import torchio as tio


config = dict(
    project = 'DeepFocus',
    entity = 'almadl',
    group = 'multigpu',
    name = 'multigpou_test_01',
    dataset = 'ALMA',
    inference_path = '/media/storage/big_cube/dirty_cube_0.fits',
    output_path = '/media/storage/big_cube/processed.fits',
    plot_path = '/media/storage/big_cube/plots/',
    batch_size = 64,
    dmode = 'deconvolver',
    block_sizes = [16, 32, 64, 128],
    oblock_sizes = [16, 8, 1],
    kernel_sizes = [(5, 5, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
    depths = [3, 4, 6, 3],
    output_kernel_sizes = [3, 3, 3],
    hidden_size = 256,
    encoder_activation = 'leaky_relu',
    decoder_activation = 'leaky_relu',
    block = 'basic', # basic, bottleneck
    skip_connections = True,
    in_channels = 1,
    out_channels = 1,
    input_shape = (256, 256, 128),
    patch_overlap = (8, 8, 8),
    final_activation = 'sigmoid',
    multi_gpu = True,
    multi_node = False, 
    log_rate = 1,
    save_frequency = 1,
    num_workers = 4, 
    num_gpus = 2,
    num_nodes = 1,
    debug = False,
)

if __name__ == "__main__":
    dl.inference_multigpu(config)
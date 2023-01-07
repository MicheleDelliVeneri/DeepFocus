import pprint
import wandb
import utils.dl_utils as dl
import torchio as tio
wandb.login()

config = dict(
    project = 'ALMA3D',
    entity = 'almadl',
    name = 'basic_skip_3d_01',
    get_tclean = True,
    resume = False,
    dataset = 'ALMA',
    data_path = '/lustre/home/mdelliveneri/ALMADL/data/',
    output_path = '/lustre/home/mdelliveneri/TNGDL/saved_models/',
    plot_path = '/lustre/home/mdelliveneri/TNGDL/plots/',
    tclean_path = '/lustre/home/mdelliveneri/TNGDL/tcleaned/',
    epochs = 100,
    dmode = 'deconvolver', # deconvolver, regressor, classifier
    parameter = 'flux', # x, y, flux, continuum, fwhm_x, fwhm_y, pa
    num_workers =  8, 
    weight_decay = 0.0001,
    learning_rate = 0.0001,
    dropout_rate = 0.0,
    batch_size = 8,
    optimizer = 'Adam',
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
    criterion= ['L1', 'SSIM'],
    normalize = True,
    final_activation = 'sigmoid',
    warm_start = True,
    warm_start_iterations = 10,
    debug=False,
    log_rate = 25,
    preprocess = 'log',
)


dl.train(config)
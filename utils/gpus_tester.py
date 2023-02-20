import wandb
wandb.login()



config = dict(
    project = 'DeepFocus',
    entity = 'almadl',
    name = 'multigpu_3d_01',
    dataset = 'ALMA',
    data_path = '/ibiscostorage/mdelliveneri/ALMA/data/',
    output_path = '/ibiscostorage/mdelliveneri/ALMA/saved_models/',
    plot_path = '/ibiscostorage/mdelliveneri/ALMA/plots/',
    epochs = 20,
    weight_decay = 0.0001,
    learning_rate = 0.0001,
    dropout_rate = 0.0,
    batch_size = 32,
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
    multi_gpu = True,
    log_rate = 1,
    num_workers = 4, 
    num_gpus = 2,
    num_nodes = 1,
)


wandb_run = wandb.init(project=config['project'], entity=config['entity'], name=config['name'], config=config)
config = wandb_run.config
run_id = wandb_run.id
print(run_id)
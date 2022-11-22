import utils.model_utils as mut
import torch 
import pandas as pd
import os 
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# DO NOT RUN THIS FILE!!!! BUT YOU CAN COPY AND PASTE THE CODE TO YOUR MAIN FILE
# First of all setup weight and biases
wandb.login()
# Examples of single models training 
# ----------------------------------------------------------------

# Training Blobs Finder 
hyperparams = dict(
    epochs = 200,
    batch_size = 64, 
    muli_gpu = False, 
    mode = 'train', 
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    hidden_channels = 1024,
    criterion=['l_1', 'ssim'],
    warm_start=True,
    warm_start_iterations = 10,
    detection_threshold = 0.15,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model_name = 'blobsfinder',
    experiment_name = 'blobsfinder_test_01',
    entity = 'mdelliveneri',
)
with wandb.init(project=hyperparams['project'], name=hyperparams['experiment_name'], entity=hyperparams['entity'], config=hyperparams):
    config = wandb.config
    blobsfinder, criterion, optimizer, train_loader, valid_loader = mut.make_blobsfinder(config, device)
    mut.train(blobsfinder, criterion, optimizer, train_loader, valid_loader)



# Training Deep GRU
config = dict(
    epochs = 200,
    batch_size = 64, 
    muli_gpu = False, 
    mode = 'train', 
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    warm_start=False,
    warm_start_iterations = 3,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model_name = 'deepgru',
    experiment_name = 'deepgru_test_01',
    entity = 'mdelliveneri',
)
with wandb.init(project=hyperparams['project'], name=hyperparams['experiment_name'], entity=hyperparams['entity'], config=hyperparams):
    config = wandb.config
    deepgru, criterion, optimizer, train_loader, valid_loader = mut.make_deepgru(config, device)
    mut.train(deepgru, criterion, optimizer, train_loader, valid_loader)

# Training the ResNets
# FWHM X
config = dict(
    epochs = 200,
    batch_size = 64, 
    muli_gpu = False, 
    mode = 'train', 
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    hidden_channels = 1024,
    warm_start=False,
    warm_start_iterations = 3,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model_name = 'fwhmx_resnet',
    param = 'fwhm_x',
    experiment_name = 'fwhm_x_renet_test_01',
    entity = 'mdelliveneri',
)
with wandb.init(project=hyperparams['project'], name=hyperparams['experiment_name'], entity=hyperparams['entity'], config=hyperparams):
    config = wandb.config
    fwhmx_resnet, criterion, optimizer, train_loader, valid_loader = mut.make_resnet(config, device)
    mut.train(fwhmx_resnet, criterion, optimizer, train_loader, valid_loader)

# FWHM Y
config = dict(
    epochs = 200,
    batch_size = 64, 
    muli_gpu = False, 
    mode = 'train', 
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    hidden_channels = 1024,
    warm_start=False,
    warm_start_iterations = 10,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model_name = 'fwhmy_resnet',
    param = 'fwhm_y',
    experiment_name = 'fwhm_y_renet_test_01',
    entity = 'mdelliveneri',
)
with wandb.init(project=hyperparams['project'], name=hyperparams['experiment_name'], entity=hyperparams['entity'], config=hyperparams):
    config = wandb.config
    fwhmy_resnet, criterion, optimizer, train_loader, valid_loader = mut.make_resnet(config, device)
    mut.train(fwhmy_resnet, criterion, optimizer, train_loader, valid_loader)

# PA
config = dict(
    epochs = 200,
    batch_size = 64, 
    muli_gpu = False, 
    mode = 'train', 
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    hidden_channels = 1024,
    warm_start=False,
    warm_start_iterations = 10,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model_name = 'pa_resnet',
    param = 'pa',
    experiment_name = 'pa_renet_test_01',
    entity = 'mdelliveneri',
)
pa_resnet, criterion, optimizer, train_loader, valid_loader = mut.make_resnet(config, device)
mut.train(pa_resnet, criterion, optimizer, train_loader, valid_loader)

# FLUX
config = dict(
    epochs = 200,
    batch_size = 64, 
    muli_gpu = False, 
    mode = 'train', 
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    hidden_channels = 1024,
    warm_start=False,
    warm_start_iterations = 10,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model_name = 'flux_resnet',
    param = 'flux',
    experiment_name = 'flux_renet_test_01',
    entity = 'mdelliveneri',
    
)
with wandb.init(project=hyperparams['project'], name=hyperparams['experiment_name'], entity=hyperparams['entity'], config=hyperparams):
    config = wandb.config
    fwhmx_resnet, criterion, optimizer, train_loader, valid_loader = mut.make_resnet(config, device)
    mut.train(fwhmx_resnet, criterion, optimizer, train_loader, valid_loader)

# Train each model on the predictions of the previous one
# ------------------------------------------------------------------------------------------------
config = dict(
    epochs = 50,
    batch_size = 64, 
    muli_gpu = False, 
    mode = 'train', 
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    warm_start=False,
    warm_start_iterations = 10,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    blobsfinder_name = 'blobsfinder',
    deepGRU_name = 'deepGRU',
    fwhmx_resnet_name = 'fwhmx_resnet',
    fwhmy_resnet_name = 'fwhmy_resnet',
    pa_resnet_name = 'pa_resnet',
    flux_resnet_name = 'flux_resnet'
)
mut.train_on_predictions(config, device)


# Set the pipeline in inference mode and run predictions
#----------------------------------------------------------------
predictions = mut.run_pipeline(config, device)
pd.to_csv(os.path.join(config['prediction_dir'], 'predictions.csv'))

import utils.model_utils as mut
import torch 
import pandas as pd
import os 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Examples of single models training 
# ----------------------------------------------------------------

# Training Blobs Finder 
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
    criterion=['l_1', 'ssim'],
    warm_start=True,
    warm_start_iterations = 10,
    detection_threshold = 0.15,
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'ALMA',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model_name = 'blobsfinder'
)

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
    model_name = 'deepgru'
)
deepgru, criterion, optimizer, train_loader, valid_loader = mut.make_blobsfinder(config, device)
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
    param = 'fwhm_x'
)
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
    param = 'fwhm_y'
)
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
    param = 'pa'
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
    param = 'flux'
)
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
# Now that the model are trained, produce predictions on the test set 
#----------------------------------------------------------------
predictions = mut.run_pipeline(config, device)
pd.to_csv(os.path.join(config['prediction_dir'], 'predictions.csv'))

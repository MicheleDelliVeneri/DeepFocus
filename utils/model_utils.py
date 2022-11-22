from tkinter import W
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from torchvision import transforms
from torchmetrics import MeanSquaredLogError
import numpy as np
import pandas as pd
import models.resnet as rn
import models.blobsfinder as bf
import models.deepgru as dg
import os
import wandb
import random
import utils.load_data as ld
from tqdm import tqdm
import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib
from skimage.feature import blob_dog
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.measure import regionprops, label
from kornia.losses import SSIMLoss
from scipy.signal import find_peaks
from astropy.modeling import models, fitting
from photutils.aperture import CircularAnnulus, CircularAperture
matplotlib.rcParams.update({'font.size': 12})
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("setting random seeds") % 2**32 - 1)
torch.manual_seed(hash("setting random seeds") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("setting random seeds") % 2**32 - 1)

def save_checkpoint(model, optimizer, save_path, epoch):
    """
    This function saves the model state_dict, the optimizer
    state dict and the epoch.
    Inputs:
    model: the model to be saved;
    optimizer: the optimizer to be saved;
    save_path: the path to save the checkpoint;
    epoch: the current epoch number.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        save_path,
    )

def load_checkpoint(model, optimizer, load_path):
    """
    This function loads the model state_dict, the optimizer state
    dict and the epoch.
    Inputs:
    model: the model;
    optimizer: the optimizer;
    load_path: the path to load the checkpoint;
    """
    checkpoint = torch.load(load_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def make_blobsfinder(config, device):
    """
    This function creates a blob finder object based on the config, loads the data, 
    and creates the data loaders. If the mode in the config dictionary is 
    set to 'train', then the function outputs the trainin and validation 
    data loaders, else the function outputs the test dataloader. 
    Inputs:
    config: the config file used to build the blob finder.
    device: the device used to run the blob finder.
    Outputs:
    blob_finder: the blob finder object;
    criterion: the loss function;
    optimizer: the optimizer;
    data_loaders: the data loaders. 

    """
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    train_dir = config['data_folder'] + 'Train/'
    valid_dir = config['data_folder'] + 'Validation/'
    test_dir = config['data_folder'] + 'Test/'
    crop = ld.Crop(256)
    rotate = ld.RandomRotate()
    hflip = ld.RandomHorizontalFlip(p=1)
    vflip = ld.RandomVerticalFlip(p=1)
    norm_img = ld.NormalizeImage()
    to_tensor = ld.ToTensor(model='blobsfinder')
    train_compose = transforms.Compose([rotate, vflip, hflip, crop, norm_img, to_tensor])
    if config['mode'] == 'train':
        print('Preparing Data for Blobs Finder Training and Testing...')
        train_dataset = ld.ALMADataset('train_params.csv', train_dir, transform=train_compose)
        valid_dataset = ld.ALMADataset('valid_params.csv', valid_dir, transform=train_compose)
    
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count() // 3, 
                          pin_memory=True, shuffle=True, collate_fn=train_dataset.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count() // 3, 
                          pin_memory=True, shuffle=True, collate_fn=valid_dataset.collate_fn)
    else:
        print("Preparing Data for Blobs Finder Testing....")
    test_compose = transforms.Compose([crop, norm_img, to_tensor])
    test_dataset = ld.ALMADataset('test_params.csv', test_dir, transform=test_compose)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count() // 3, 
                          pin_memory=True, shuffle=False, collate_fn=test_dataset.collate_fn)
    model = bf.BlobsFinder(config['hidden_channels'])
    if torch.cuda.device_count() > 1 and config['multi_gpu']:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    model.to(device)
    criterion_name = config['criterion']
    if len(criterion_name) > 0:
        criterion = []
        for crit in criterion_name:
            if crit == 'l_1':
                criterion.append(nn.L1Loss())
            elif crit == 'l_2':
                criterion.append(nn.MSELoss())
            elif crit == 'ssim':
                criterion.append(SSIMLoss(window_size=3))
    else:
        if criterion_name == 'l_1':
            criterion = nn.L1Loss()
        elif criterion_name == 'l_2':
            criterion = nn.MSELoss()
        elif criterion_name == 'ssim':
            criterion = SSIMLoss(window_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    if config['mode'] == 'train':
        return model, criterion, optimizer, train_loader, valid_loader
    else:
        outpath = os.sep.join((config['output_dir'], config['model_name'] + ".pt"))
        model, _, _ = load_checkpoint(model, optimizer, outpath)
        return model, criterion, optimizer, test_loader

def make_deepgru(config, device):
    """
    This function creates a deepgru object based on the config, loads the data, 
    and creates the data loaders. If the mode in the config dictionary is 
    set to 'train', then the function outputs the trainin and validation 
    data loaders, else the function outputs the test dataloader. 
    Inputs:
    config: the config file used to build the blob finder.
    device: the device used to run the blob finder.
    Outputs:
    blob_finder: the blob finder object;
    criterion: the loss function;
    optimizer: the optimizer;
    data_loaders: the data loaders. 

    """
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    train_dir = config['data_folder'] + 'Train/'
    valid_dir = config['data_folder'] + 'Validation/'
    test_dir = config['data_folder'] + 'Test/'
    if config['mode'] == 'train':
        traindata = ld.PipelineDataLoader('train_params.csv', train_dir)
        validdata = ld.PipelineDataLoader('valid_params.csv', valid_dir)
        t_spectra, t_dspectra, t_focused, t_targets, t_line_images = traindata.create_dataset()
        v_spectra, v_dspectra, v_focused, v_targets, v_line_images = validdata.create_dataset()
    testddata = ld.PipelineDataLoader('test_params.csv', test_dir)
    te_spectra, te_dspectra, te_focused, te_targets, te_line_images = testddata.create_dataset()
    if config['mode'] == 'train':
        train_dataset = TensorDataset(torch.Tensor(t_dspectra), torch.Tensor(t_spectra))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                          pin_memory=True, shuffle=True)
        valid_dataset = TensorDataset(torch.Tensor(v_dspectra), torch.Tensor(v_spectra))
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                          pin_memory=True, shuffle=True)
    test_dataset = TensorDataset(torch.Tensor(te_dspectra), torch.Tensor(te_spectra))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(),
                                  pin_memory=True, shuffle=False)
    model = dg.DeepGRU(1, 32, 1, True)
    if torch.cuda.device_count() > 1 and config['multi_gpu']:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    print(f'Using {device}') 
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    if config['mode'] == 'train':
        return model, criterion, optimizer, train_loader, valid_loader
    else:
        outpath = os.sep.join((config['output_dir'], config['model_name'] + ".pt"))
        model, _, _ = load_checkpoint(model, optimizer, outpath)
        return model, criterion, optimizer, test_loader

def make_resnet(config, device):
    """
    This function creates a ResNet object based on the config, loads the data, 
    and creates the data loaders. If the mode in the config dictionary is 
    set to 'train', then the function outputs the trainin and validation 
    data loaders, else the function outputs the test dataloader. 
    Inputs:
    config: the config file used to build the blob finder.
    device: the device used to run the blob finder.
    Outputs:
    blob_finder: the blob finder object;
    criterion: the loss function;
    optimizer: the optimizer;
    data_loaders: the data loaders. 

    """
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    train_dir = config['data_folder'] + 'Train/'
    valid_dir = config['data_folder'] + 'Validation/'
    test_dir = config['data_folder'] + 'Test/'
    if config['mode'] == 'train':
        traindata = ld.PipelineDataLoader('train_params.csv', train_dir)
        validdata = ld.PipelineDataLoader('valid_params.csv', valid_dir)
        t_spectra, t_dspectra, t_focused, t_targets, t_line_images = traindata.create_dataset()
        v_spectra, v_dspectra, v_focused, v_targets, v_line_images = validdata.create_dataset()
    testddata = ld.PipelineDataLoader('test_params.csv', test_dir)
    te_spectra, te_dspectra, te_focused, te_targets, te_line_images = testddata.create_dataset()
    if config['mode'] == 'train':
        if config['param'] == 'flux':
            train_dataset = TensorDataset(torch.Tensor(t_line_images), torch.Tensor(t_targets))
            valid_dataset = TensorDataset(torch.Tensor(v_line_images), torch.Tensor(v_targets))
        else:
            train_dataset = TensorDataset(torch.Tensor(t_focused), torch.Tensor(t_targets))
            valid_dataset = TensorDataset(torch.Tensor(v_focused), torch.Tensor(v_targets))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                        pin_memory=True, shuffle=True)
                
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                        pin_memory=True, shuffle=True)
    else:
        test_dataset = TensorDataset(torch.Tensor(te_line_images), torch.Tensor(te_targets))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(),
                                  pin_memory=True, shuffle=False)
    model = rn.ResNet18(1, 1)
    if torch.cuda.device_count() > 1 and config['multi_gpu']:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    print(f'Using {device}') 
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    if config['mode'] == 'train':
        return model, criterion, optimizer, train_loader, valid_loader
    else:
        outpath = os.sep.join((config['output_dir'], config['model_name'] + ".pt"))
        model, _, _ = load_checkpoint(model, optimizer, outpath)
        return model, criterion, optimizer, test_loader
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Inputs:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer, save_path, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            save_checkpoint(model, optimizer, save_path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_checkpoint(model, optimizer, save_path, epoch)
            self.counter = 0

def train_batch(inputs, targets, model, optimizer, criterion):
    """
    Batch training wrapper function, makes both the forward and 
    backward passes through the model to compute the loss and update 
    the model weights.
    Inputs:
    inputs: the input tensor;
    targets: the target tensor;
    model: the model to train;
    optimizer: the optimizer;
    criterion: the loss function.

    Outputs:
    loss: the loss value;
    outputs: the output tensor;

    """
    optimizer.zero_grad()
    outputs = model(inputs)
    if isinstance(criterion, list):
        loss = 0
        for i in range(len(criterion)):
            loss += criterion[i](outputs, targets)
    else:
        loss = criterion(outputs, targets)  
    loss.backward()
    optimizer.step()
    return loss, outputs

def valid_batch(inputs, targets, model, optimizer, criterion):
    """
    Batch validation wrapper function, makes only the forward pass
    through the model to compute the loss.
    Inputs:
    inputs: the input tensor;
    targets: the target tensor;
    model: the model to train;
    optimizer: the optimizer;
    criterion: the loss function.

    Outputs:
    loss: the loss value;
    outputs: the output tensor;

    """
    optimizer.zero_grad()
    outputs = model(inputs)
    if isinstance(criterion, list):
        loss = 0
        for i in range(len(criterion)):
            loss += criterion[i](outputs, targets)
    else:
        loss = criterion(outputs, targets)
    
    return loss, outputs

def test_batch(inputs, targets, model, criterion):
    """
    Batch test wrapper function, makes only the forward pass 
    through the model to compute loss.
    Inputs:
    inputs: the input tensor;
    targets: the target tensor;
    model: the model to train;
    optimizer: the optimizer;
    criterion: the loss function.

    Outputs:
    loss: the loss value;
    outputs: the output tensor;

    """
    outputs = model(inputs)
    if isinstance(criterion, list):
        loss = 0
        for i in range(len(criterion)):
            loss += criterion[i](outputs, targets)
    else:
        loss = criterion(outputs, targets)
    return loss, outputs

def train_log(loss, optimizer, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "train_loss": loss, 'learning_rate': optimizer.param_groups[0]['lr']})

def valid_log(loss):
    # Where the magic happens
    wandb.log({"valid_loss": loss})

def test_log(loss):
    wandb.log({"test_loss": loss})

def log_images(inputs, predictions, targets, mode='Train'):
    """
    This function logs the Blobs Finder images to wandb for visual inspection
    """
    idxs  = random.sample(list(np.arange(0, len(predictions))), 8)
    inputs_log = inputs[idxs]
    predictions_log = predictions[idxs]
    targets_log = targets[idxs]
    images = torch.cat((inputs_log, predictions_log, targets_log), dim=0)
    images = make_grid(images, nrow=8,  normalize=True)
    imgs = wandb.Image(images, caption="Top: Inputs, Center: Predictions, Bottom: Targets")
    if mode == 'Train':
        wandb.log({"train_examples": imgs})
    elif mode == 'Validation':
        wandb.log({"validation_examples": imgs})
    else:
        wandb.log({"test_examples": imgs})

def log_parameters(outputs, targets, mode, config):
    """
    This function logs the ResNet prediction vs truth scatter plots
    on wandb for visual inspection
    """

    data = [[x, y] for (x, y) in zip(targets[:, 0].cpu().detach().numpy(), outputs[:, 0].cpu().detach().numpy())]
    data = wandb.Table(data=data, columns = ["Target", "Prediction"])
    if mode=='Train':
        wandb.log({'Training_plot': wandb.plot.scatter(data, x='Target {}'.format(config['param']), 
                            y='Predicted {}'.format(config['param']),
                            title='Training {} Scatter Plot'.format(config['param']))})
    
    else:
        wandb.log({'Validation_plot': wandb.plot.scatter(data, x='Target {}'.format(config['param']), 
                            y='Predicted {}'.format(config['param']),
                            title='Validation {} Scatter Plot'.format(config['param']))})

def log_spectra(inputs, predictions, targets, mode='Train'):
    """
    This function logs the DeepGRU predicted spectra on wandb for visual inspection.
    """
    idxs  = random.sample(list(np.arange(0, len(predictions))), 8)
    inputs_log = inputs[idxs].cpu().detach().numpy()
    predictions_log = predictions[idxs].cpu().detach().numpy()
    targets_log = targets[idxs].cpu().detach().numpy()
    fig, ax = plt.subplots(nrows=3, ncols=8, figsize=(8 * 8, 4 * 3))
    for i in range(len(inputs_log)):
        ax[0, i].plot(inputs_log[i, :, 0], label='Input Dirty Spectrum')
        ax[0, i].set_xlabel('Frequency')
        ax[0, i].set_ylabel('Amplitude')
        ax[1, i].plot(predictions_log[i, :, 0], label='Predicted Clean Spectrum')
        ax[1, i].set_xlabel('Frequency')
        ax[1, i].set_ylabel('Amplitude')
        ax[2, i].plot(targets_log[i, :, 0], label='Target Clean Spectrum')
        ax[2, i].set_xlabel('Frequency')
        ax[2, i].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.legend()
    if mode == 'Train':
        wandb.log({"train_examples": fig})
    elif mode == 'Validation':
        wandb.log({"validation_examples": fig})
    else:
        wandb.log({"test_examples": fig})

def param_selector(y, param):
    """
    This function is used to select the right parameter for each ResNet
    Inputs:
    y: input tensor containing all the target parameters for the batch;
    param: name of the parameter to select.
    Returns:
    the selected parameter.
    """   
    if param == 'fwhm_x':
        return y[:, 0][:, None]
    elif param == 'fwhm_y':
        return y[:, 1][:, None]
    elif param == 'pa':
        return y[:, 2][:, None]
    elif param == 'flux':
        return y[:, 3][:, None]

def normalize_spectra(y):
    """
    This function is used to normalize the spectra of a batch
    Inputs:
    y: input spectra;
    Returns:
    the normalized spectra.
    """
    for b in range(len(y)):
        t = y[b, :, 0]
        t = (t - torch.min(t)) / (torch.max(t) - torch.min(t))
        y[b, :, 0] = t
    return y

def oned_iou(a, b):
    """
    This function penforms the 1D IoU between two vectors a and b
    """
    assert a[0] <= a[1]
    assert b[0] <= b[1]
    x_l = max(a[0], b[0])
    x_r = min(a[1], b[1])
    if x_l > x_r:
        return 0.0
    intr = (x_r - x_l)
    la = a[1] - a[0]
    lb = b[1] - b[0]
    iou = intr / (la + lb - intr)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def train(model, train_loader, valid_loader, criterion, 
          optimizer, config, name, device):
    example_ct = 0
    best_loss = 9999
    # initialize the early_stopping object
    if config['early_stopping']:
        early_stopping = EarlyStopping(patience=config['patience'], verbose=False)
    outpath = os.sep.join((config['output_dir'], name + ".pt"))
    for epoch in tqdm(range(config.epochs)):
        iteration_counter = 0
        model.train()
        running_loss = 0.0
        nb = len(train_loader)
        for i_batch, batch in tqdm(enumerate(train_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]
            if config['model'] == 'resnet':
                targets = param_selector(targets, config['param']).to(device)
            targets.to(device)
            if config['model'] == 'spectral':
                inputs = normalize_spectra(inputs)
                targets = normalize_spectra(targets)
            iteration_number = iteration_counter + epoch * nb
            if config['warm_start'] and iteration_number < config['warm_start_iterations']:
                xi = [0, config['warm_start_iterations']]
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(iteration_number, xi, [config['warm_start_lr'], config['lr']])
            loss, outputs = train_batch(inputs, targets, model, optimizer, criterion)
            example_ct += len(inputs)
            train_log(loss, optimizer, epoch)
            if i_batch == len(train_loader) - 1:
                if config['model'] == 'blobsfinder':
                    log_images(inputs, outputs, targets, 'Train')
                if config['model'] == 'spectral':
                    log_spectra(inputs, outputs, targets, 'Train')
                if config['model'] == 'resnet':
                    log_parameters(outputs, targets, 'Train', config)
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss {epoch_loss}")
        torch.cuda.empty_cache()
        model.eval()
        running_loss = 0.0
        valid_losses = []
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(valid_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                if config['model'] == 'resnet':
                    targets = param_selector(targets, config['param']).to(device)
                    targets.to(device)
                if config['model'] == 'spectral':
                    inputs = normalize_spectra(inputs)
                    targets = normalize_spectra(targets)
                loss, outputs = valid_batch(inputs, targets, model, optimizer, criterion)
                valid_log(loss)
                if config['model'] == 'blobsfinder':
                    log_images(inputs, outputs, targets, 'Validation')
                if config['model'] == 'spectral':
                    log_spectra(inputs, outputs, targets, 'Validation')
                if config['model'] == 'resnet':
                    log_parameters(outputs, targets, 'Validation', config)
                running_loss += loss.item() * inputs.size(0)
                valid_losses.append(loss.item())
        valid_loss = np.average(valid_losses)
        epoch_loss = running_loss / len(valid_loader.dataset)
        print(f"Validation Loss {epoch_loss}")
        if config['early_stopping']:
            early_stopping(valid_loss, model, optimizer, outpath, epoch)
        else:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(model, optimizer, outpath , epoch)
    model, _, _ = load_checkpoint(model, optimizer, outpath)
    return model

def remove_diag(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    out = strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)
    return out

def test(model, test_loader, criterion, config, device):
    model.eval()
    if not os.path.exists(config['plot_dir']):
        os.mkdir(config['plot_dir'])
    if not os.path.exists(config['prediction_dir']):
        os.mkdir(config['prediction_dir'])
    tp, fp, fn = 0
    t_x, t_y, p_x, p_y = [], [], [], []
    t_z, p_z, t_extension, p_extension, = [], [], [], []
    pIoUs, tIoUs, pfluxes, tfluxes, pSNRs, tSNRs = [], [], [], [], [], []
    tparameters, pparameters = [], []
    for i_batch, batch in tqdm(enumerate(test_loader)):
        inputs = batch[0].to(device)
        targets = batch[1]
        if config['model'] == 'resnet':
            targets = param_selector(targets, config['param']).to(device)
        targets.to(device)
        if config['model'] == 'spectral':
            inputs = normalize_spectra(inputs)
            targets = normalize_spectra(targets)
        if config['model'] == 'blobsfinder':
            target_boxes =  batch[2]
            snrs = batch[3]
            target_parameters = batch[4]
        loss, outputs = test_batch(inputs, targets, model, criterion)
        if config['model'] == 'blobsfinder':
            for b in tqdm(range(len(targets))):
                tboxes = target_boxes[b]
                tsnrs = snrs[b]
                fluxes = target_parameters[b][:, 5]
                tboxes_ious = np.max(remove_diag(box_iou(torch.Tensor(tboxes), torch.Tensor(tboxes)).numpy()), axis=1)
                output = outputs[b, 0].cpu().detach().numpy()
                min_, max_ = np.min(output), np.max(output)
                output = (output - min_) / (max_ - min_)
                seg = output.copy()
                seg[seg >= config['detection_threshold']] = 1
                seg = seg.astype(int)
                struct = generate_binary_structure(2, 2)
                seg = binary_dilation(seg, struct)
                props = regionprops(label(seg, connectivity=2))
                boxes = []
                for prop in props:
                    y0, x0, y1, x1 = prop.bbox
                    boxes.append([y0, x0, y1, x1])
                boxes = np.array(boxes)
                txc = tboxes[:, 1] + 0.5 * (tboxes[:, 3] - tboxes[:, 1])
                tyc = tboxes[:, 0] + 0.5 * (tboxes[:, 2] - tboxes[:, 0])
                pxs = boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1])
                pys = boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0])
                # merasuring distances and IoUs between true and predicted bounding boxes
                dists = []
                for j in range(len(txc)):
                    d = []
                    for k in range(len(pxs)):
                        d.append(np.sqrt((txc[j] - pxs[k])**2 + (tyc[j] - pys[k])**2))
                    dists.append(d)
                dists = np.array(dists)
                idxs = np.argmin(dists, axis=1)
                dists = np.min(dists, axis=1)
                ious = box_iou(torch.Tensor(tboxes), torch.Tensor(boxes)).numpy()
                ious = np.max(ious, axis=1)
                for i in range(len(dists)):
                    if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                        t_x.append(txc[i])
                        t_y.append(tyc[i])
                        p_x.append(pxs[idxs[i]])
                        p_y.append(pys[idxs[i]])
                        tIoUs.append(tboxes_ious[i])
                        tfluxes.append(fluxes[i])
                        pSNRs.append(tsnrs[i])
                        tp += 1
                    else:
                        fn += 1
                    tIoUs.append(tboxes_ious[i])
                    tSNRs.append(tsnrs[i])
                    tfluxes.append(fluxes[i])
    
                if len(boxes) > len(tboxes):
                    fp += len(boxes) - len(tboxes)
            pSNRs = np.array(pSNRs)
            tSNRs = np.array(tSNRs)
            t_x = np.array(t_x)
            t_y = np.array(t_y)
            p_x = np.array(p_x)
            p_y = np.array(p_y)
            tIoUs = np.array(tIoUs)
            pIoUs = np.array(pIoUs)
            tfluxes = np.array(tfluxes)
            pfluxes = np.array(pfluxes)
            pxname = os.path.join(config['prediction_dir'], 'predicted_x.npy')
            pyname = os.path.join(config['prediction_dir'], 'predicted_y.npy')
            txname = os.path.join(config['prediction_dir'], 'true_x.npy')
            tyname = os.path.join(config['prediction_dir'], 'true_y.npy')
            np.save(pxname, p_x)
            np.save(pyname, p_y)
            np.save(txname, t_x)
            np.save(tyname, t_y)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return tp, fp, fn, precision, recall, len(test_loader.dataset)
        if config['model'] == 'spectral':
            for b in range(len(outputs)):
                # normalizing spectra
                tspectrum = targets[b, :, 0].cpu().detach().numpy()
                pspectrum = outputs[b, :, 0].cpu().detach().numpy()
                min_, max_ = np.min(tspectrum), np.max(tspectrum)
                ty = (tspectrum - min_) / (max_ - min_)
                min_, max_ = np.min(pspectrum), np.max(pspectrum)
                y = (pspectrum - min_) / (max_ - min_)
                x = np.array(range(len(y)))
                # finding peaks in outputs
                peaks, _ = find_peaks(y, height=np.mean(y) + 0.1, prominence=0.05, distance=10)
                peaks_amp = y[peaks]
                x_peaks = x[peaks]
                # finding peaks in targets
                tpeaks, _ = find_peaks(ty, height=0.0, prominence=0.05, distance=10)
                tpeaks_amp = ty[tpeaks]
                tx_peaks = x[tpeaks]
                lims, idxs = [], []
                #fitting target and output peaks
                fit_g = fitting.LevMarLSQFitter()
                for i_peak, peak in enumerate(x_peaks):
                    g1 = models.Gaussian1D(amplitude=peaks_amp[i_peak], mean=peak, stddev=3)
                    if peak > 10 and peak < 118:
                        g = fit_g(g1, x[peak - 10: peak + 10], y[peak - 10: peak + 10])
                    elif peak <= 10:
                        g = fit_g(g1, x[:peak + 10], y[:peak + 10])
                    else:
                        g = fit_g(g1, x[peak - 10:], y[peak - 10:])
                    m, dm = int(g.mean.value), int(g.fwhm)
                    if dm < 64:
                        lims.append([int(x_peaks[i_peak]) - dm, int(x_peaks[i_peak]) + dm])
                        idxs.append(i_peak)
                tlims, tidxs = [], []
                for i_peak, peak in enumerate(tx_peaks):
                    g1 = models.Gaussian1D(amplitude=tpeaks_amp[i_peak], mean=peak, stddev=3)
                    if peak > 10 and peak < 118:
                        g = fit_g(g1, x[peak - 10: peak + 10], ty[peak - 10: peak + 10])
                    elif peak <= 10:
                        g = fit_g(g1, x[:peak + 10], ty[:peak + 10])
                    else:
                        g = fit_g(g1, x[peak - 10:], ty[peak - 10:])
                    m, dm = int(g.mean.value), int(g.fwhm)
                    if dm < 64:
                        tlims.append([int(tx_peaks[i_peak]) - dm, int(tx_peaks[i_peak]) + dm])
                        tidxs.append(i_peak)
                if len(lims) > 0 and len(tlims) > 0:
                    #computing distances between peaks
                    dists = []
                    ious = []
                    for j in range(len(tx_peaks)):
                        d = []
                        i = []
                        for k in range(len(x_peaks)):
                            d.append(np.sqrt((tx_peaks[j] - x_peaks[k]) ** 2))
                            i.append(oned_iou(tlims[j], lims[k]))
                        dists.append(d)
                        ious.append(i)
                    dists = np.array(dists)
                    ious = np.array(ious)
                    ious = np.max(ious, axis=1)
                    dists = np.min(dists, axis=1)
                    for i in range(len(dists)):
                        if dists[i] <= config['oneD_dist_threshold'] and ious[i] >= config['oneD_iou_threshold']:
                            t_z.append(tx_peaks[i])
                            p_z.append(x_peaks[i])
                            t_extension.append(tlims[i][1] - tlims[i][0])
                            p_extension.append(lims[i][1] - lims[i][0])
                            tp += 1
                        else:
                            fn += 1
                    if len(lims) > len(tlims):
                        fp += len(lims) - len(tlims)
            t_z = np.array(t_z)
            p_z = np.array(p_z)
            t_extension = np.array(t_extension)
            p_extension = np.array(p_extension)
            pzname = os.path.join(config['prediction_dir'], 'predicted_z.npy')
            pextname = os.path.join(config['prediction_dir'], 'predicted_ext.npy')
            tzname = os.path.join(config['prediction_dir'], 'true_z.npy')
            textname = os.path.join(config['prediction_dir'], 'true_ext.npy')
            np.save(pzname, p_z)
            np.save(pextname, p_extension)
            np.save(tzname, t_z)
            np.save(textname, t_extension)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return tp, fp, fn, precision, recall, len(test_loader.dataset)
        
        if config['model'] == 'resnet':
            for b in range(len(outputs)):
                target = targets[b].cpu().detach().numpy()
                prediction = outputs[b].cpu().detach().numpy()
                for i in range(len(target)):
                    tparameters.append(target[i])
                    pparameters.append(prediction[i])
            tparameters = np.array(tparameters)
            pparameters = np.array(pparameters)
            if config['param'] == 'fwhm_x':
                pname = os.path.join(config['prediction_dir'], 'predicted_fwhmx.npy')
                tname = os.path.join(config['prediction_dir'], 'true_fwhmx.npy')
            elif config['param'] == 'fwhm_y':
                pname = os.path.join(config['prediction_dir'], 'predicted_fwhmy.npy')
                tname = os.path.join(config['prediction_dir'], 'true_fwhmy.npy')
            elif config['param'] == 'pa':
                pname = os.path.join(config['prediction_dir'], 'predicted_pa.npy')
                tname = os.path.join(config['prediction_dir'], 'true_pa.npy')
            elif config['param'] == 'flux':
                pname = os.path.join(config['prediction_dir'], 'predicted_flux.npy')
                tname = os.path.join(config['prediction_dir'], 'true_flux.npy')
            np.save(pname, pparameters)
            np.save(tname, tparameters)
            return pparameters, tparameters

def get_spectra_from_dataloader(config, blobsfinder, bf_criterion, loader, device):
    """
    This function runs Blobs Finder on data and returns input and dirty spectra extracted from its predictions. 
    Inputs:
        config: model configuration dictionary
        blobsfinder: BlobsFinder instance
        bf_criterion: loss function to be used for computing the loss
        loader: torch.utils.data.DataLoader object
        device: the device on which to run the model
    Outputs:
        loader: torch.utils.data.DataLoader object containg the spectra. 
    """
    
    blobsfinder.eval()
    spectra = []
    dspectra = []
    for i_batch, batch in tqdm(enumerate(loader)):
        inputs = batch[0].to(device)
        targets = batch[1]
        dirty_cubes = batch[5]
        clean_cubes = batch[6]
        loss, outputs = test_batch(inputs, targets, blobsfinder, bf_criterion)
        for b in tqdm(range(len(outputs))):
            output = outputs[b, 0].cpu().detach().numpy()
            dirty_cube = dirty_cubes[b, 0]
            clean_cube = clean_cubes[b, 0]
            min_, max_ = np.min(output), np.max(output)
            output = (output - min_) / (max_ - min_)
            seg = output.copy()
            seg[seg >= config['detection_threshold']] = 1
            seg = seg.astype(int)
            struct = generate_binary_structure(2, 2)
            seg = binary_dilation(seg, struct)
            props = regionprops(label(seg, connectivity=2))
            boxes = []
            for prop in props:
                y0, x0, y1, x1 = prop.bbox
                boxes.append([y0, x0, y1, x1])
            boxes = np.array(boxes)
            dirty_spectra = np.array([
                np.sum(dirty_cube[:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])
            clean_spectra = np.array([
                np.sum(clean_cube[:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])
            for j in range(len(dirty_spectra)):
                dspec = dirty_spectra[j]
                dspec = (dspec - np.mean(dspec)) / np.std(dspec)
                spec = clean_spectra[j]
                spec = (spec - np.mean(spec)) / np.std(spec)
                spectra.append(spec)
                dspectra.append(dspec)
    spectra = np.array(spectra)
    dspectra = np.array(dspectra)
    dspectra = np.transpose(dspectra[np.newaxis], 
                                        axes=(1, 2, 0))
    spectra = np.transpose(spectra[np.newaxis], 
                                        axes=(1, 2, 0)) 
    dataset = TensorDataset(torch.Tensor(dspectra), torch.Tensor(spectra))
    loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                          pin_memory=True, shuffle=True)
    return loader

def localSNR(img):
    "This function computes the per pixel SNR map"
    return np.abs(img / (np.std(img) **2))

def globalSNR(img, box):
    """
    This function computes the Global SNR
    """
    y0, x0, y1, x1 = box
    xc, yc = img.shape[0] // 2,  img.shape[1] // 2
    r0, r1 = 1.6 * (x1 - x0), 2.6 * (x1 - x0)
    r = 0.5 * (x1 - x0)
    noise_aperture = CircularAnnulus((xc, yc), r0 / 2, r1 / 2 )
    mask = noise_aperture.to_mask(method='center')
    source_aperture = CircularAperture((xc, yc), r)
    aperture_mask = source_aperture.to_mask()
    noise_p = mask.multiply(img)
    noise_p = noise_p[mask.data > 0]
    source_p = aperture_mask.multiply(img)
    source_p = source_p[aperture_mask.data > 0.]
    var = np.std(noise_p) ** 2
    mean = np.median(source_p)
    snr = np.abs(mean / var)
    return snr

def get_focussed_from_dataloader(config, blobsfinder, deepGRU, bf_criterion, dg_criterion, loader, device):
    blobsfinder.eval()
    deepGRU.eval()
    focussed = []
    parameters = []
    for i_batch, batch in tqdm(enumerate(loader)):
        inputs = batch[0].to(device)
        targets = batch[1]
        params = batch[4]
        dirty_cubes = batch[5]
        clean_cubes = batch[6]
        loss, outputs = test_batch(inputs, targets, blobsfinder, bf_criterion)
        spectra = []
        dspectra = []
        for b in tqdm(range(len(outputs))):
            output = outputs[b, 0].cpu().detach().numpy()
            param = params[b]
            dirty_cube = dirty_cubes[b, 0]
            clean_cube = clean_cubes[b, 0]
            min_, max_ = np.min(output), np.max(output)
            output = (output - min_) / (max_ - min_)
            seg = output.copy()
            seg[seg >= config['detection_threshold']] = 1
            seg = seg.astype(int)
            struct = generate_binary_structure(2, 2)
            seg = binary_dilation(seg, struct)
            props = regionprops(label(seg, connectivity=2))
            boxes = []
            for prop in props:
                y0, x0, y1, x1 = prop.bbox
                boxes.append([y0, x0, y1, x1])
            boxes = np.array(boxes)
            dirty_spectra = np.array([
                np.sum(dirty_cube[:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])
            clean_spectra = np.array([
                np.sum(clean_cube[:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])
            for j in range(len(dirty_spectra)):
                parameters.append(param[j])
                dspec = dirty_spectra[j]
                dspec = (dspec - np.mean(dspec)) / np.std(dspec)
                spec = clean_spectra[j]
                spec = (spec - np.mean(spec)) / np.std(spec)
                spectra.append(spec)
                dspectra.append(dspec)
        spectra = np.array(spectra)
        dspectra = np.array(dspectra)
        dspectra = torch.Tensor(np.transpose(dspectra[np.newaxis], 
                                        axes=(1, 2, 0)))
        spectra = torch.Tensor(np.transpose(spectra[np.newaxis], 
                                        axes=(1, 2, 0)))
        loss, outputs = test_batch(dspectra, spectra, deepGRU, dg_criterion)
        for b in tqdm(range(len(outputs))):
            y_0, x_0, y_1, x_1  = boxes[b]
            width_x, width_y = x_1 - x_0, y_1 - y_0
            x, y = x_0 + 0.5 * width_x, y_0 + 0.5 * width_y
            spectrum = outputs[b, :, 0].cpu().detach().numpy()
            dirty_cube = dirty_cubes[b, 0]
            min_, max_ = np.min(spectrum), np.max(spectrum)
            y = (spectrum - min_) / (max_ - min_)
            x = np.array(range(len(y)))
            peaks, _ = find_peaks(y, height=np.mean(y) + 0.1, prominence=0.05, distance=10)
            x_peaks = x[peaks]
            peaks_amp = y[peaks]
            # peaks are sorted by amplitude
            idxs = np.argsort(-peaks_amp)
            peaks_amp = peaks_amp[idxs]
            x_peaks = x_peaks[idxs]

            
            lims, idxs = [], []
            #fitting target and output peaks
            fit_g = fitting.LevMarLSQFitter()
            for i_peak, peak in enumerate(x_peaks):
                g1 = models.Gaussian1D(amplitude=peaks_amp[i_peak], mean=peak, stddev=3)
                if peak > 10 and peak < 118:
                    g = fit_g(g1, x[peak - 10: peak + 10], y[peak - 10: peak + 10])
                elif peak <= 10:
                    g = fit_g(g1, x[:peak + 10], y[:peak + 10])
                else:
                    g = fit_g(g1, x[peak - 10:], y[peak - 10:])
                m, dm = int(g.mean.value), int(g.fwhm)
                if dm < 64:
                    lims.append([int(x_peaks[i_peak]) - dm, int(x_peaks[i_peak]) + dm])
                    idxs.append(i_peak)
            
            # Source Focusing and SNR reasoning

            # the first and brightest source is focused and checks are made to retain or discard
            source = np.sum(dirty_cube[lims[0][0]: lims[0][1], int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
            reference = np.sum(dirty_cube[:, int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
            xsize, ysize = int(source.shape[0]), int(source.shape[1])
            dx, dy = xsize - 64, ysize - 64
            if dx % 2 == 0:
                left, right = dx // 2, xsize - dx // 2
            else:
                left, right = dx // 2, xsize - dx // 2 - 1
            if dy % 2 == 0:
                bottom, top = dy // 2, ysize - dy // 2
            else:
                bottom, top = dy // 2, ysize - dy // 2 - 1
            source = source[left:right, bottom:top]
            reference = reference[left:right, bottom:top]
            gsourceSNR = globalSNR(source, boxes[b])
            greferenceSNR = globalSNR(reference, boxes[b])
            lprimarySNR = localSNR(source)
            #detect highest SNR pixel in the image
            rpix = np.argmax(lprimarySNR)
            if gsourceSNR >= 6:
                min_, max_ = np.min(source), np.max(source)
                source = (source - min_) / (max_ - min_)
                focussed.append(source[np.newaxis])
            else:
                if gsourceSNR >= greferenceSNR:
                    min_, max_ = np.min(source), np.max(source)
                    source = (source - min_) / (max_ - min_)
                    focussed.append(source[np.newaxis])
            
            # if there are secondary peaks, then
            if len(lims) > 1:
                for i_peak in range(len(lims))[1:]:
                    source = np.sum(dirty_cube[lims[i_peak][0]: lims[i_peak][1], int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                    reference = np.sum(dirty_cube[:, int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                    xsize, ysize = int(source.shape[0]), int(source.shape[1])
                    dx, dy = xsize - 64, ysize - 64
                    if dx % 2 == 0:
                        left, right = dx // 2, xsize - dx // 2
                    else:
                        left, right = dx // 2, xsize - dx // 2 - 1
                    if dy % 2 == 0:
                        bottom, top = dy // 2, ysize - dy // 2
                    else:
                        bottom, top = dy // 2, ysize - dy // 2 - 1
                    source = source[left:right, bottom:top]
                    reference = reference[left:right, bottom:top]
                    gsourceSNR = globalSNR(source, boxes[b])
                    greferenceSNR = globalSNR(reference, boxes[b])
                    lSNR = localSNR(source)
                    pix = np.argmax(lSNR)
                    if pix != rpix and gsourceSNR >= greferenceSNR:
                        min_, max_ = np.min(source), np.max(source)
                        source = (source - min_) / (max_ - min_)
                        focussed.append(source[np.newaxis])
    focussed = np.array(focussed)
    parameters = np.array(parameters)
    dataset = TensorDataset(torch.Tensor(focussed), torch.Tensor(parameters))   
    loader =  DataLoader(dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                        pin_memory=True, shuffle=False)
    return loader

def train_on_predictions(config, device):
    """
    This function train each model on the predictions from the previous ones.
    Inputs:
    config: the dictionary containing the configuration parameters;
    device: the name of the device to use for training the models;

    """
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # loading images for BlobsFinder
    train_dir = config['data_folder'] + 'Train/'
    valid_dir = config['data_folder'] + 'Validation/'
    test_dir = config['data_folder'] + 'Test/'
    crop = ld.Crop(256)
    to_tensor = ld.ToTensor(model='blobsfinder')
    norm_img = ld.NormalizeImage()
    compose = transforms.Compose([crop, norm_img, to_tensor])
    train_dataset = ld.ALMADataset('train_params.csv', train_dir, transform=compose)
    valid_dataset = ld.ALMADataset('valid_params.csv', valid_dir, transform=compose)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count() // 3, 
                          pin_memory=True, shuffle=False, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count() // 3, 
                          pin_memory=True, shuffle=False, collate_fn=valid_dataset.collate_fn)
    print('Loading all Deep Learning Models')
    # Loading Blobs Finder latest checkpoint
    blobsfinder = bf.BlobsFinder(config['hidden_channels'])
    bf_optimizer = torch.optim.Adam(blobsfinder.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    bf_criterion = [nn.SSIMLoss(), nn.L1Loss()]
    if torch.cuda.device_count() > 1 and config['multi_gpu']:
        print(f'Using {torch.cuda.device_count()} GPUs')
        blobsfinder = nn.DataParallel(blobsfinder)
    blobsfinder.to(device)

    outpath = os.sep.join((config['output_dir'], config['blobsfinder_name'] + ".pt"))
    blobsfinder, _, _ = load_checkpoint(blobsfinder, bf_optimizer, outpath)
    
    # Loading Deep GRU and the other ResNets
    deepGRU = dg.DeepGRU(1, 32, 1, True).to(device)
    dg_criterion = nn.L1Loss()
    dg_optimizer = torch.optim.Adam(deepGRU.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    outpath = os.sep.join((config['output_dir'], config['deepGRU_name'] + ".pt"))
    deepGRU, _, _ = load_checkpoint(deepGRU, dg_optimizer, outpath)

    # Loading ResNets
    resnet = rn.ResNet18(1, 1).to(device)
    resnet_optimizer = torch.optim.Adam(resnet.parameters, lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    fwhmx_outpath = os.sep.join((config['output_dir'], config['fwhmx_resnet_name'] + ".pt"))
    fwhmx_resnet = load_checkpoint(resnet, resnet_optimizer, fwhmx_outpath)
    fwhmy_outpath = os.sep.join((config['output_dir'], config['fwhmy_resnet_name'] + ".pt"))
    fwhmy_resnet = load_checkpoint(resnet, resnet_optimizer, fwhmy_outpath)
    pa_outpath = os.sep.join((config['output_dir'], config['pa_resnet_name'] + ".pt"))
    pa_resnet = load_checkpoint(resnet, resnet_optimizer, pa_outpath)
    flux_outpath = os.sep.join((config['output_dir'], config['flux_resnet_name'] + ".pt"))
    flux_resnet = load_checkpoint(resnet, resnet_optimizer, flux_outpath)
    resnet_criterion = nn.L1Loss()
    
    # Running Blobsfinder on train and validation loaders to get predictions
    print('Getting Blobs Finder predictions on Train and Validation sets..')
    train_spectral_loader = get_spectra_from_dataloader(config, blobsfinder, bf_criterion, train_loader, device)
    valid_spectral_loader = get_spectra_from_dataloader(config, blobsfinder, bf_criterion, valid_loader, device)
    print('Training Deep GRU on Blobs Finder prediction for 50 iterations....')
    config.epochs = 50
    deepGRU = train(deepGRU, train_spectral_loader, valid_spectral_loader, 
                    dg_criterion, dg_optimizer, config, 'deepGRU_BFPreds', device)
    train_resnet_loader = get_focussed_from_dataloader(config, blobsfinder, deepGRU, bf_criterion, dg_criterion, 
                            train_loader, device)
    valid_resnet_loader = get_focussed_from_dataloader(config, blobsfinder, deepGRU, bf_criterion, dg_criterion, 
                            valid_loader, device)
    config['param'] = 'fwhm_x'
    fwhmx_resnet = train(fwhmx_resnet, train_resnet_loader, valid_resnet_loader, resnet_criterion, 
                    resnet_optimizer, config, 'resnet_fwhmx_deepGRUpreds', device)
    config['param'] = 'fwhm_y'
    fwhmy_resnet = train(fwhmy_resnet, train_resnet_loader, valid_resnet_loader, resnet_criterion, 
                    resnet_optimizer, config, 'resnet_fwhmy_deepGRUpreds', device)
    config['param'] = 'pa'
    pa_resnet = train(pa_resnet, train_resnet_loader, valid_resnet_loader, resnet_criterion, 
                    resnet_optimizer, config, 'resnet_pa_deepGRUpreds', device)
    config['param'] = 'flux'
    flux_resnet = train(flux_resnet, train_resnet_loader, valid_resnet_loader, resnet_criterion, 
                    resnet_optimizer, config, 'resnet_flux_deepGRUpreds', device)

def run_pipeline(config, device):
    predictions = []
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    test_dir = config['data_folder'] + 'Test/'
    crop = ld.Crop(256)
    to_tensor = ld.ToTensor(model='blobsfinder')
    norm_img = ld.NormalizeImage()
    compose = transforms.Compose([crop, norm_img, to_tensor])
    test_loader = ld.ALMADataset('test_params.csv', test_dir, transform=compose)
    print('Loading all Deep Learning Models')
    # Loading Blobs Finder latest checkpoint
    blobsfinder = bf.BlobsFinder(config['hidden_channels'])
    bf_optimizer = torch.optim.Adam(blobsfinder.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    bf_criterion = [nn.SSIMLoss(), nn.L1Loss()]
    if torch.cuda.device_count() > 1 and config['multi_gpu']:
        print(f'Using {torch.cuda.device_count()} GPUs')
        blobsfinder = nn.DataParallel(blobsfinder)
    blobsfinder.to(device)
    outpath = os.sep.join((config['output_dir'], config['blobsfinder_name'] + ".pt"))
    blobsfinder, _, _ = load_checkpoint(blobsfinder, bf_optimizer, outpath)
    # Loading Deep GRU and the other ResNets
    deepGRU = dg.DeepGRU(1, 32, 1, True).to(device)
    dg_criterion = nn.L1Loss()
    dg_optimizer = torch.optim.Adam(deepGRU.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    outpath = os.sep.join((config['output_dir'], config['deepGRU_name'] + ".pt"))
    deepGRU, _, _ = load_checkpoint(deepGRU, dg_optimizer, outpath)

    # Loading ResNets
    resnet = rn.ResNet18(1, 1).to(device)
    resnet_optimizer = torch.optim.Adam(resnet.parameters, lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    fwhmx_outpath = os.sep.join((config['output_dir'], config['fwhmx_resnet_name'] + ".pt"))
    fwhmx_resnet = load_checkpoint(resnet, resnet_optimizer, fwhmx_outpath)
    fwhmy_outpath = os.sep.join((config['output_dir'], config['fwhmy_resnet_name'] + ".pt"))
    fwhmy_resnet = load_checkpoint(resnet, resnet_optimizer, fwhmy_outpath)
    pa_outpath = os.sep.join((config['output_dir'], config['pa_resnet_name'] + ".pt"))
    pa_resnet = load_checkpoint(resnet, resnet_optimizer, pa_outpath)
    flux_outpath = os.sep.join((config['output_dir'], config['flux_resnet_name'] + ".pt"))
    flux_resnet = load_checkpoint(resnet, resnet_optimizer, flux_outpath)
    resnet_criterion = nn.L1Loss()
    # setting all models in eval mode
    blobsfinder.eval()
    deepGRU.eval()
    fwhmx_resnet.eval()
    fwhmy_resnet.eval()
    pa_resnet.eval()
    flux_resnet.eval()
    cube_id = 0

    tp, fp, fn = 0
    
    pIoUs, tIoUs, pfluxes, tfluxes, pSNRs, tSNRs = [], [], [], [], [], []
    tparameters, pparameters = [], []


    for i_batch, batch in tqdm(enumerate(test_loader)):
        inputs = batch[0].to(device)
        targets = batch[1]
        params = batch[4]
        target_boxes = batch[2]
        dirty_cubes = batch[5]
        clean_cubes = batch[6]
        loss, bf_outputs = test_batch(inputs, targets, blobsfinder, bf_criterion)
        spectra = []
        dspectra = []
        parameters = []
        focussed = []
        ids = []
        for b in tqdm(range(len(bf_outputs))):
            output = bf_outputs[b, 0].cpu().detach().numpy()
            param = params[b]
            tboxes = target_boxes[b]
            dirty_cube = dirty_cubes[b, 0]
            clean_cube = clean_cubes[b, 0]
            min_, max_ = np.min(output), np.max(output)
            output = (output - min_) / (max_ - min_)
            seg = output.copy()
            seg[seg >= config['detection_threshold']] = 1
            seg = seg.astype(int)
            struct = generate_binary_structure(2, 2)
            seg = binary_dilation(seg, struct)
            props = regionprops(label(seg, connectivity=2))
            boxes = []
            for prop in props:
                y0, x0, y1, x1 = prop.bbox
                boxes.append([y0, x0, y1, x1])
            boxes = np.array(boxes)
            dirty_spectra = np.array([
                np.sum(dirty_cube[:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])
            clean_spectra = np.array([
                np.sum(clean_cube[:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])
            for j in range(len(dirty_spectra)):
                parameters.append(param[j])
                dspec = dirty_spectra[j]
                dspec = (dspec - np.mean(dspec)) / np.std(dspec)
                spec = clean_spectra[j]
                spec = (spec - np.mean(spec)) / np.std(spec)
                spectra.append(spec)
                dspectra.append(dspec)
            
            ids = np.array(ids)
            spectra = np.array(spectra)
            dspectra = np.array(dspectra)
            dspectra = torch.Tensor(np.transpose(dspectra[np.newaxis], 
                                        axes=(1, 2, 0)))
            spectra = torch.Tensor(np.transpose(spectra[np.newaxis], 
                                        axes=(1, 2, 0)))
            parameters = np.array(parameters)
            dspectra = torch.Tensor(np.transpose(dspectra[np.newaxis], 
                                        axes=(1, 2, 0)))
            spectra = torch.Tensor(np.transpose(spectra[np.newaxis], 
                                        axes=(1, 2, 0)))
            dspectra = normalize_spectra(dspectra)
            spectra = normalize_spectra(spectra)
            loss, dg_outputs = test_batch(dspectra, spectra, deepGRU, dg_criterion)
            sboxes, slims, sx_peaks = [], [], []
            for bg in tqdm(range(len(dg_outputs))):
                y_0, x_0, y_1, x_1  = boxes[bg]
                width_x, width_y = x_1 - x_0, y_1 - y_0
                x, y = x_0 + 0.5 * width_x, y_0 + 0.5 * width_y
                spectrum = dg_outputs[bg, :, 0].cpu().detach().numpy()
                dirty_cube = dirty_cubes[b, 0]
                min_, max_ = np.min(spectrum), np.max(spectrum)
                y = (spectrum - min_) / (max_ - min_)
                x = np.array(range(len(y)))
                peaks, _ = find_peaks(y, height=np.mean(y) + 0.1, prominence=0.05, distance=10)
                x_peaks = x[peaks]
                peaks_amp = y[peaks]
                # peaks are sorted by amplitude
                idxs = np.argsort(-peaks_amp)
                peaks_amp = peaks_amp[idxs]
                x_peaks = x_peaks[idxs]
                lims, idxs = [], []
                #fitting target and output peaks
                fit_g = fitting.LevMarLSQFitter()
                for i_peak, peak in enumerate(x_peaks):
                    g1 = models.Gaussian1D(amplitude=peaks_amp[i_peak], mean=peak, stddev=3)
                    if peak > 10 and peak < 118:
                        g = fit_g(g1, x[peak - 10: peak + 10], y[peak - 10: peak + 10])
                    elif peak <= 10:
                        g = fit_g(g1, x[:peak + 10], y[:peak + 10])
                    else:
                        g = fit_g(g1, x[peak - 10:], y[peak - 10:])
                    m, dm = int(g.mean.value), int(g.fwhm)
                    if dm < 64:
                        lims.append([int(x_peaks[i_peak]) - dm, int(x_peaks[i_peak]) + dm])
                        idxs.append(i_peak)
                # Source Focusing and SNR reasoning
                # the first and brightest source is focused and checks are made to retain or discard
                source = np.sum(dirty_cube[lims[0][0]: lims[0][1], int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                reference = np.sum(dirty_cube[:, int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                xsize, ysize = int(source.shape[0]), int(source.shape[1])
                dx, dy = xsize - 64, ysize - 64
                if dx % 2 == 0:
                    left, right = dx // 2, xsize - dx // 2
                else:
                    left, right = dx // 2, xsize - dx // 2 - 1
                if dy % 2 == 0:
                    bottom, top = dy // 2, ysize - dy // 2
                else:
                    bottom, top = dy // 2, ysize - dy // 2 - 1
                source = source[left:right, bottom:top]
                reference = reference[left:right, bottom:top]
                gsourceSNR = globalSNR(source, boxes[b])
                greferenceSNR = globalSNR(reference, boxes[b])
                lprimarySNR = localSNR(source)
                #detect highest SNR pixel in the image
                rpix = np.argmax(lprimarySNR)
                if gsourceSNR >= 6:
                    min_, max_ = np.min(source), np.max(source)
                    source = (source - min_) / (max_ - min_)
                    sboxes.append(extract_box(source, int(x), int(y), config))
                    focussed.append(source[np.newaxis])
                    slims.append(lims[0])
                    sx_peaks.append(x_peaks[0])
                else:
                    if gsourceSNR >= greferenceSNR:
                        min_, max_ = np.min(source), np.max(source)
                        source = (source - min_) / (max_ - min_)
                        focussed.append(source[np.newaxis])
                        sboxes.append(extract_box(source, int(x), int(y), config))
                        slims.append(lims[0])
                        sx_peaks.append(x_peaks[0])
                # if there are secondary peaks, then
                if len(lims) > 1:
                    for i_peak in range(len(lims))[1:]:
                        source = np.sum(dirty_cube[lims[i_peak][0]: lims[i_peak][1], int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                        reference = np.sum(dirty_cube[:, int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                        xsize, ysize = int(source.shape[0]), int(source.shape[1])
                        dx, dy = xsize - 64, ysize - 64
                        if dx % 2 == 0:
                            left, right = dx // 2, xsize - dx // 2
                        else:
                            left, right = dx // 2, xsize - dx // 2 - 1
                        if dy % 2 == 0:
                            bottom, top = dy // 2, ysize - dy // 2
                        else:
                            bottom, top = dy // 2, ysize - dy // 2 - 1
                        source = source[left:right, bottom:top]
                        reference = reference[left:right, bottom:top]
                        gsourceSNR = globalSNR(source, boxes[bg])
                        greferenceSNR = globalSNR(reference, boxes[bg])
                        lSNR = localSNR(source)
                        pix = np.argmax(lSNR)
                        if pix != rpix and gsourceSNR >= greferenceSNR:
                            min_, max_ = np.min(source), np.max(source)
                            source = (source - min_) / (max_ - min_)
                            focussed.append(source[np.newaxis])
                            sboxes.append(extract_box(source, int(x), int(y), config))
                            slims.append(lims[i_peak])
                            sx_peaks.append(x_peaks[i_peak])


                # extracting truth parameters for spectral direction
                tspectrum = spectra[b, 0]
                min_, max_ = np.min(tspectrum), np.max(tspectrum)
                ty = (tspectrum - min_) / (max_ - min_)
                x = np.array(range(len(ty)))
                tpeaks, _ = find_peaks(ty, height=0.0, prominence=0.05, distance=10)
                tpeaks_amp = ty[tpeaks]
                tx_peaks = x[tpeaks]
                tlims, tidxs = [], []
                for i_peak, peak in enumerate(tx_peaks):
                    g1 = models.Gaussian1D(amplitude=tpeaks_amp[i_peak], mean=peak, stddev=3)
                    if peak > 10 and peak < 118:
                        g = fit_g(g1, x[peak - 10: peak + 10], ty[peak - 10: peak + 10])
                    elif peak <= 10:
                        g = fit_g(g1, x[:peak + 10], ty[:peak + 10])
                    else:
                        g = fit_g(g1, x[peak - 10:], ty[peak - 10:])
                    m, dm = int(g.mean.value), int(g.fwhm)
                    if dm < 64:
                        tlims.append([int(tx_peaks[i_peak]) - dm, int(tx_peaks[i_peak]) + dm])
                        tidxs.append(i_peak)
            
            sboxes = np.array(sboxes)
            sx_peaks = np.array(sx_peaks)
            slims = np.array(slims)   
            focussed = torch.Tensor(np.array(focussed))
            parameters = torch.Tensor(parameters)
            
            # creating positions for surving boundin boxes and getting true bounding boxes
            pxs = sboxes[:, 1] + 0.5 * (sboxes[:, 3] - sboxes[:, 1])
            pys = sboxes[:, 0] + 0.5 * (sboxes[:, 2] - sboxes[:, 0])
            txc = tboxes[:, 1] + 0.5 * (tboxes[:, 3] - tboxes[:, 1])
            tyc = tboxes[:, 0] + 0.5 * (tboxes[:, 2] - tboxes[:, 0])
            tlims = np.array(tlims)
            tidxs = np.array(tidxs)

            t_x, t_y, p_x, p_y = [], [], [], []
            t_z, p_z, t_extension, p_extension, = [], [], [], []
            stx_peaks = []
            stlims = []
            # Checking matching criteria
            # spatial criterium
            dists = []
            for j in range(len(txc)):
                d = []
                for k in range(len(pxs)):
                    d.append(np.sqrt((txc[j] - pxs[k])**2 + (tyc[j] - pys[k])**2))
                dists.append(d)
                dists = np.array(dists)
                idxs = np.argmin(dists, axis=1)
                dists = np.min(dists, axis=1)
                ious = box_iou(torch.Tensor(tboxes), torch.Tensor(boxes)).numpy()
                ious = np.max(ious, axis=1)
                for i in range(len(dists)):
                    if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                        t_x.append(txc[i])
                        t_y.append(tyc[i])
                        p_x.append(pxs[idxs[i]])
                        p_y.append(pys[idxs[i]])
                        stx_peaks.append(sx_peaks[idxs[i]])
                        stlims.append(slims[idxs[i]])
                    else:
                         fn += 1
            stx_peaks = np.array(stx_peaks)
            stlims = np.array(stlims)
            # frequency criterium
            dists = []
            ious = []
            for j in range(len(tx_peaks)):
                dists.append(np.sqrt((tx_peaks[j] - stx_peaks[j]) ** 2))
                ious.append(oned_iou(tlims[j], stlims[j]))
            dists = np.array(dists)
            ious = np.array(ious)
            sidxs = []
            for i in range(len(dists)):
                if dists[i] <= config['oneD_dist_threshold'] and ious[i] >= config['oneD_iou_threshold']:
                    t_z.append(tx_peaks[i])
                    p_z.append(x_peaks[i])
                    t_extension.append(tlims[i][1] - tlims[i][0])
                    p_extension.append(lims[i][1] - lims[i][0])
                    tp += 1
                    sidxs.append(i)
                else:
                    fn += 1

            
            t_x = np.array(t_x)[sidxs]
            t_y = np.array(t_y)[sidxs]
            p_x = np.array(p_x)[sidxs]
            p_y = np.array(p_y)[sidxs]
            t_z = np.array(t_z)
            p_z = np.array(p_z)
            t_extension = np.array(t_extension)
            s_extension = np.array(s_extension)
            
            focussed = focussed[sidxs]
            parameters = parameters[sidxs]
            loss, fwhmxs = test_batch(focussed, parameters, fwhmx_resnet, resnet_criterion)
            loss, fwhmys = test_batch(focussed, parameters, fwhmy_resnet, resnet_criterion)
            loss, pas = test_batch(focussed, parameters, pa_resnet, resnet_criterion)
            loss, fluxes = test_batch(focussed, parameters, flux_resnet, resnet_criterion)

            ids = np.ones(len(p_x)) * cube_id
            cube_id += 1
            pparams = np.column_stack((ids, p_x, p_y, p_z, fwhmxs, fwhmys, s_extension, pas, fluxes))
        predictions.append(pparams)
        
    predictions = np.concatenate(predictions, axis=0)

    columns = ['id', 'x', 'y', 'z', 'fwhm_x', 'fwhm_y', 'dz' 'pa', 'flux']
    db = pd.DataFrame(predictions, columns=columns)
    return db

def extract_box(image, dx, dy, config):
    seg = image.copy()
    seg[seg >= config['detection_threshold']] = 1
    seg = seg.astype(int)
    struct = generate_binary_structure(2, 2)
    seg = binary_dilation(seg, struct)
    props = regionprops(label(seg, connectivity=2))
    boxes = []
    for prop in props:
        y0, x0, y1, x1 = prop.bbox
        boxes.append([y0, x0, y1, x1])
    box = np.array(boxes)[0]
    box[1] = box[1] + dx
    box[3] = box[3] + dx
    box[0] = box[0] + dy
    box[2] = box[2] + dy
    return box



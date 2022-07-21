import numpy as np
import pandas as pd
import os
from pytest import param
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import utils.load_data as ld
from tqdm import tqdm
import torch
import random
from kornia.losses import SSIMLoss
from torchvision.ops import box_iou
from torchvision.utils import make_grid
from skimage.measure import regionprops, label
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib 
from scipy.ndimage import binary_dilation
from astropy.modeling import models, fitting
from scipy import ndimage
import matplotlib.pyplot as plt
import models.resnet as rn
import models.blobsfinder as bf
import models.deepgru as dg
import matplotlib
from astropy.io import fits
from torch.utils.data import TensorDataset, DataLoader
matplotlib.rcParams.update({'font.size': 12})
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("setting random seeds") % 2**32 - 1)
torch.manual_seed(hash("setting random seeds") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("setting random seeds") % 2**32 - 1)

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        save_path,
    )

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
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
            save_checkpoint(val_loss, model, optimizer, save_path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_checkpoint(val_loss, model, optimizer, save_path, epoch)
            self.counter = 0

def train_batch(inputs, targets, model, optimizer, criterion):
    outputs = model(inputs)
    if isinstance(criterion, list):
        loss = 0
        for i in range(len(criterion)):
            loss += criterion[i](outputs, targets)
    else:
        loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, outputs

def valid_batch(inputs, targets, model, optimizer, criterion):
    outputs = model(inputs)
    if isinstance(criterion, list):
        loss = 0
        for i in range(len(criterion)):
            loss += criterion[i](outputs, targets)
    else:
        loss = criterion(outputs, targets)
    optimizer.zero_grad()
    return loss, outputs

def test_batch(inputs, targets, model, criterion):
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

def param_selector(y, param):
    if param == 'x':
        return y[:, 0][:, None]
    elif param == 'y':
        return y[:, 1][:, None]
    elif param == 'fwhm_x':
        return y[:, 2][:, None]
    elif param == 'fwhm_y':
        return y[:, 3][:, None]
    elif param == 'pa':
        return y[:, 4][:, None]
    elif param == 'flux':
        return y[:, 5][:, None]
    elif param == 'continuum':
        return y[:, 6][:, None]

def normalize_spectra(y):
    for b in range(len(y)):
        t = y[b, :, 0]
        t = (t - torch.min(t)) / (torch.max(t) - torch.min(t))
        y[b, :, 0] = t
    return y

def iou(a, b):
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

def log_parameters(outputs, targets, mode, config):
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

def log_images(inputs, predictions, targets, mode='Train'):
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

def log_spectra(inputs, predictions, targets, mode='Train'):
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

def train(model, train_loader, valid_loader, criterion, optimizer, config, device):
    example_ct = 0
    best_loss = 9999
    # initialize the early_stopping object
    if config['early_stopping']:
        early_stopping = EarlyStopping(patience=config['patience'], verbose=False)

    outpath = os.sep.join((config['output_dir'], config['name'] + ".pt"))
    for epoch in tqdm(range(config.epochs)):
        model.train()
        running_loss = 0.0
        for i_batch, batch in tqdm(enumerate(train_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]
            if config['model'] == 'resnet':
                targets = param_selector(targets, config['param']).to(device)
            targets.to(device)
            if config['model'] == 'spectral':
                inputs = normalize_spectra(inputs)
                targets = normalize_spectra(targets)

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
                if i_batch == len(train_loader) - 1:
                    if config['model'] == 'blobsfinder':
                        log_images(inputs, outputs, targets, 'Train')
                    if config['model'] == 'spectral':
                        log_spectra(inputs, outputs, targets, 'Train')
                    if config['model'] == 'resnet':
                        log_parameters(outputs, targets, 'Train', config)
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

def test(model, test_loader, criterion, config, device):
    model.eval()
    t = 0
    fp = 0
    if not os.path.exists(config['plot_dir']):
        os.mkdir(config['plot_dir'])
    if not os.path.exists(config['prediction_dir']):
        os.mkdir(config['prediction_dir'])
    tgs = []
    pds = []
    true_x = []
    true_y = []
    predicted_x = []
    predicted_y = []
    true_z = []
    predicted_z = []
    true_extension = []
    predicted_extension = []
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
        targets = targets.to(device)
        loss, outputs = test_batch(inputs, targets, model, criterion)
        test_log(loss)
        if config['model'] == 'blobsfinder':
            for b in tqdm(range(len(targets))):
                output = outputs[b, 0].cpu().detach().numpy()
                min_, max_ = np.min(output), np.max(output)
                output = (output - min_) / (max_ - min_)
                tboxes = target_boxes[b]
                seg = output.copy()
                seg[seg > 0.15] = 1
                seg = seg.astype(int)
                struct = ndimage.generate_binary_structure(2, 2)
                seg = binary_dilation(seg, struct)
                props = regionprops(label(seg, connectivity=2))
                boxes = []
                for prop in props:
                    y0, x0, y1, x1 = prop.bbox
                    boxes.append([y0, x0, y1, x1])
                boxes = np.array(boxes)
                ious = box_iou(torch.Tensor(tboxes), torch.Tensor(boxes)).numpy()
                ious = np.max(ious, axis=1)
                txc = tboxes[:, 1] + 0.5 * (tboxes[:, 3] - tboxes[:, 1])
                tyc = tboxes[:, 0] + 0.5 * (tboxes[:, 2] - tboxes[:, 0])
                xc = boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1])
                yc = boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0])
                dists = []
                #dists = [[np.sqrt((txc[j] - xc[k])**2 + (tyc[j] - yc[k])**2 for k in range(len(xc))] for j in range(len(txc))]
                for j in range(len(txc)):
                    d = []
                    for k in range(len(xc)):
                        d.append(np.sqrt((txc[j] - xc[k])**2 + (tyc[j] - yc[k])**2))
                    dists.append(d)
                dists = np.array(dists)
                idxs = np.argmin(dists, axis=1)
                dists = np.min(dists, axis=1)
                
                for i in range(len(dists)):
                    if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                        true_x.append(txc[i])
                        true_y.append(tyc[i])
                        predicted_x.append(xc[idxs[i]])
                        predicted_y.append(yc[idxs[i]])
                        t += 1
                if len(boxes) > len(tboxes):
                    fp += len(boxes) - len(tboxes)
                    

        if config['model'] == 'spectral':
            for b in range(len(outputs)):
                tspectrum = targets[b, 3:127, 0].cpu().detach().numpy()
                pspectrum = targets[b, 3:127, 0].cpu().detach().numpy()
                min_, max_ = np.min(pspectrum), np.max(pspectrum)
                tmin_, tmax_ = np.min(tspectrum), np.max(tspectrum)
                y = (pspectrum - min_) / (max_ - min_)
                ty = (tspectrum -tmin_) / (tmax_ - tmin_)
                x = np.array(range(len(y)))
                peaks, _ = find_peaks(y, height=np.mean(y) + 0.1, prominence=0.05, distance=10)
                peaks_amp = y[peaks]
                x_peaks = x[peaks]
                tpeaks, _ = find_peaks(ty, height=0.0, prominence=0.05, distance=10)
                tpeaks_amp = ty[tpeaks]
                tx_peaks = x[tpeaks]
                lims = []
                idxs = []
                for i_peak, peak in enumerate(x_peaks):
                    g1 = models.Gaussian1D(amplitude=peaks_amp[i_peak], mean=peak, stddev=3)
                    fit_g = fitting.LevMarLSQFitter()
                    if peak >= 10 and peak <= 118:
                        g = fit_g(g1, x[peak - 10: peak + 10], y[peak - 10: peak + 10])
                    elif peak < 10:
                        g = fit_g(g1, x[0: peak + 10], y[0: peak + 10])
                    else:
                        g = fit_g(g1, x[peak - 10: peak + 128 - peak], y[peak - 10: peak + 128 - peak])
                    m, dm = int(g.mean.value), int(g.fwhm)
                    if dm <= 64:        
                        lims.append([int(x_peaks[i_peak]) - dm, int(x_peaks[i_peak]) + dm])
                        idxs.append(i_peak)
                tlims = []
                tidxs = []
                for i_peak, peak in enumerate(tx_peaks):
                    g1 = models.Gaussian1D(amplitude=tpeaks_amp[i_peak], mean=peak, stddev=3)
                    fit_g = fitting.LevMarLSQFitter()
                    if peak >= 10 and peak <= 118:
                        g = fit_g(g1, x[peak - 10: peak + 10], y[peak - 10: peak + 10])
                    elif peak < 10:
                        g = fit_g(g1, x[0: peak + 10], y[0: peak + 10])
                    else:
                        g = fit_g(g1, x[peak - 10: peak + 128 - peak], y[peak - 10: peak + 128 - peak])
                    m, dm = int(g.mean.value), int(g.fwhm)
                    if dm <= 64 and int(tx_peaks[i_peak]) - dm >= 0 and int(tx_peaks[i_peak]) + dm <= 128:
                        tlims.append([int(tx_peaks[i_peak]) - dm, int(tx_peaks[i_peak]) + dm])
                        tidxs.append(i_peak)
                if len(lims) > 0:
                    x_peaks = x_peaks[idxs]
                    peaks_amp = peaks_amp[idxs]     
                if len(tlims) > 0:
                    tx_peaks = tx_peaks[tidxs]
                    tpeaks_amp = tpeaks_amp[tidxs]
                    ious = []
                
                if len(tlims) > 0 and len(lims) > 0:
                    dists = []
                    for j in range(len(tx_peaks)):
                        d = []
                        for k in range(len(x_peaks)):
                            d.append(np.sqrt((tx_peaks[j] - x_peaks[k]) ** 2))
                        dists.append(d)
                    dists = np.array(dists)
                    min_idxs = np.argmin(dists, axis=1)
                    dists = np.min(dists, axis=1)
                    for it, tlim in enumerate(tlims):
                        idx = min_idxs[it]
                        ious.append(iou(tlims[it], lims[idx]))
                    ious = np.array(ious)
                    for i in range(len(dists)):
                        if ious[i] >= config['oneD_iou_threshold'] and dists[i] <= config['oneD_dist_threshold']:
                            t += 1
                            true_z.append(tx_peaks[i])
                            predicted_z.append(x_peaks[min_idxs[i]])
                            true_extension.append(tlims[i][1] - tlims[i][0])
                            predicted_extension.append(lims[min_idxs[i]][1] -lims[min_idxs[i]][0])
    
                    if len(x_peaks) > len(tx_peaks):
                        fp += len(x_peaks) - len(tx_peaks)
        if config['model'] == 'resnet':    
            for b in range(len(outputs)):
                target = targets[b].cpu().detach().numpy()
                prediction = outputs[b].cpu().detach().numpy()
                for i in range(len(target)):
                    tgs.append(target[i])
                    pds.append(prediction[i])
    tgs = np.array(tgs)
    pds = np.array(pds)
    res = tgs - pds
    if config['model'] == 'blobsfinder':
        true_x = np.array(true_x)
        true_y = np.array(true_y)
        predicted_x = np.array(predicted_x)
        predicted_y = np.array(predicted_y)
        print(true_x.shape, true_y.shape, predicted_x.shape, predicted_y.shape)
        return t, len(test_loader.dataset), fp, true_x, true_y, predicted_x, predicted_y
    if config['model'] == 'spectral':
        true_z = np.array(true_z)
        predicted_z = np.array(predicted_z)
        true_extension = np.array(true_extension)
        predicted_extension = np.array(predicted_extension)
        return t, len(test_loader.dataset), fp, true_z, true_extension, predicted_z, predicted_extension
    if config['model'] == 'resnet':
        mean, std = np.mean(res), np.std(res)
        wandb.log({'Mean Residual': mean, 'Standard Deviation': std})
        return tgs, pds, res

def make(config, device):
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    train_dir = config['data_folder'] + 'Train/'
    valid_dir = config['data_folder'] + 'Validation/'
    test_dir = config['data_folder'] + 'Test/'
    if config['model'] == 'blobsfinder':
        crop = ld.Crop(256)
        rotate = ld.RandomRotate()
        hflip = ld.RandomHorizontalFlip(p=1)
        vflip = ld.RandomVerticalFlip(p=1)
        norm_img = ld.NormalizeImage()
        to_tensor = ld.ToTensor()
        train_compose = transforms.Compose([rotate, vflip, hflip, crop, norm_img, to_tensor])
        if config['mode'] == 'train':
            print('Preparing Data for Blobs Finder Training and Testing...')
            train_dataset = ld.ALMADataset('train_params.csv', train_dir, transform=train_compose)
            valid_dataset = ld.ALMADataset('valid_params.csv', valid_dir, transform=train_compose)
        
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=train_dataset.collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=valid_dataset.collate_fn)
        else:
            print("Preparing Data for Blobs Finder Testing....")
        test_compose = transforms.Compose([crop, norm_img, to_tensor])
        test_dataset = ld.ALMADataset('test_params.csv', test_dir, transform=test_compose)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=test_dataset.collate_fn)

        model = bf.BlobsFinder(config['input_channels'], 
                               config['blobsfinder_latent_channels'],
                               config['encoder_output_channels'],
                               config['activation_function'])
    else:
        if config['mode'] == 'train':
            traindata = ld.PipelineDataLoader('train_params.csv', train_dir)
            validdata = ld.PipelineDataLoader('valid_params.csv', valid_dir)
            t_spectra, t_dspectra, t_focused, t_targets, t_line_images = traindata.create_dataset()
            v_spectra, v_dspectra, v_focused, v_targets, v_line_images = validdata.create_dataset()

        testddata = ld.PipelineDataLoader('test_params.csv', test_dir)
        
        te_spectra, te_dspectra, te_focused, te_targets, te_line_images = testddata.create_dataset()
        if config['model'] == 'spectral':
            if config['mode'] == 'train':
                print('Preparing Data for Deep GRU Training and Testing...')
                train_dataset = TensorDataset(torch.Tensor(t_dspectra), torch.Tensor(t_spectra))
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                                  pin_memory=True, shuffle=True)
                valid_dataset = TensorDataset(torch.Tensor(v_dspectra), torch.Tensor(v_spectra))
                valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                                  pin_memory=True, shuffle=True)
            else:
                print("Preparing Data for Deep GRU Testing....")
            test_dataset = TensorDataset(torch.Tensor(te_dspectra), torch.Tensor(te_spectra))
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(),
                                  pin_memory=True, shuffle=False)
            model = dg.DeepGRU(config['input_channels'],
                               config['deepgru_latent_channels'],
                               config['output_channels'], 
                               config['bidirectional'])
        if config['model'] == 'resnet':
            if config['mode'] == 'train':
                print('Preparing Data for ResNet Training and Testing...')
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
                print("Preparing Data for ResNet Testing...")
            if config['param'] == 'flux':
                test_dataset = TensorDataset(torch.Tensor(te_line_images), torch.Tensor(te_targets))
            else:
                test_dataset = TensorDataset(torch.Tensor(te_focused), torch.Tensor(te_targets))
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(),
                                  pin_memory=True, shuffle=False)
            model = rn.ResNet18(config['input_channels'], config['output_channels'])
    if torch.cuda.device_count() > 1 and config['multi_gpu']:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    print(f'Using {device}') 
    model.to(device)
    criterion_name = config['criterion']
    if isinstance(criterion_name, list):
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
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], 
                                    momentum=0.9)
    print('Selected the ' + config['optimizer'] + ' optimizer')
    print('Selected the follwing loss function/s:')
    print(criterion_name)
    if config['mode'] == 'train':
        return model, train_loader, valid_loader, test_loader,  criterion, optimizer
    else:
        print('Loading Checkpoint....')
        outpath = os.sep.join((config['output_dir'], config['name'] + ".pt"))
        model, _, _ = load_checkpoint(model, optimizer, outpath)
        return model, test_loader, criterion, optimizer

class Pipeline(object):

    def __init__(self, hyperparameters, device):
        # folders
        output_dir = hyperparameters['output_dir']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(hyperparameters['plot_dir']):
            os.mkdir(hyperparameters['plot_dir'])
        if not os.path.exists(hyperparameters['prediction_dir']):
            os.mkdir(hyperparameters['prediction_dir'])

        self.train_dir = hyperparameters['data_folder'] + 'Train/'
        self.valid_dir = hyperparameters['data_folder'] + 'Validation/'
        self.test_dir = hyperparameters['data_folder'] + 'Test/'

        # transformations 
        crop = ld.Crop(256)
        norm_img = ld.NormalizeImage()
        to_tensor = ld.ToTensor()
        hflip = ld.RandomHorizontalFlip(p=1)
        vflip = ld.RandomVerticalFlip(p=1)
        rotate = ld.RandomRotate()
        self.train_compose = transforms.Compose([rotate, vflip, hflip, crop, norm_img, to_tensor])
        self.test_compose = transforms.Compose([crop, norm_img, to_tensor])

        # utils
        self.device = device
        self.hyperparameters = hyperparameters
        # Models Initializations
        blobsfinder = bf.BlobsFinder(self.hyperparameters['input_channels'], 
                               self.hyperparameters['blobsfinder_latent_channels'],
                               self.hyperparameters['encoder_output_channels'],
                               self.hyperparameters['activation_function'])
        deepgru = dg.DeepGRU(self.hyperparameters['input_channels'],
                               self.hyperparameters['deepgru_latent_channels'],
                               self.hyperparameters['output_channels'], 
                               self.hyperparameters['bidirectional'])
        resnet = rn.ResNet18(self.hyperparameters['input_channels'], self.hyperparameters['output_channels'])

        if torch.cuda.device_count() > 1 and self.hyperparameters['multi_gpu']:
            print(f'Using {torch.cuda.device_count()} GPUs')
            blobsfinder = nn.DataParallel(blobsfinder)
            deepgru = nn.DataParallel(deepgru)
            resnet = nn.DataParallel(resnet)
        self.blobsfinder = blobsfinder.to(self.device)
        self.deepgru = deepgru.to(self.device)
        self.resnet = resnet.to(self.device)

        # Optimizers 
        if self.hyperparameters['optimizer'] == 'Adam':
            self.blobsfinder_optimizer = torch.optim.Adam(self.blobsfinder.parameters(), lr=self.hyperparameters['blobsfinder_learning_rate'], 
                                         weight_decay=self.hyperparameters['weight_decay'])
            self.deepgru_optimizer = torch.optim.Adam(self.deepgru.parameters(), lr=self.hyperparameters['deepgru_learning_rate'], 
                                         weight_decay=self.hyperparameters['weight_decay'])
            self.resnet_optimizer = torch.optim.Adam(self.resnet.parameters(), lr=self.hyperparameters['resnet_learning_rate'], 
                                         weight_decay=self.hyperparameters['weight_decay'])
        elif self.hyperparameters['optimizer'] == 'SGD':
            self.blobsfinder_optimizer = torch.optim.SGD(self.blobsfinder.parameters(), lr=self.hyperparameters['blobsfinder_learning_rate'], 
                                    momentum=0.9)
            self.deepgru_optimizer = torch.optim.SGD(self.deepgru.parameters(), lr=self.hyperparameters['deepgru_learning_rate'], 
                                    momentum=0.9)
            self.resnet_optimizer = torch.optim.SGD(self.resnet.parameters(), lr=self.hyperparameters['resnet_learning_rate'], 
                                    momentum=0.9)

        # Criterions 
        self.blobsfinder_criterion = self.select_criterion(self.hyperparameters['blobsfinder_criterion'])
        self.deepgru_criterion = self.select_criterion(self.hyperparameters['deepgru_criterion'])
        self.resnet_criterion = self.select_criterion(self.hyperparameters['resnet_criterion'])

        # Saving and Loading paths for models
        self.blobsfinder_outpath = os.sep.join((self.hyperparameters['output_dir'], self.hyperparameters['blobsfinder_name'] + ".pt"))
        self.deepgru_outpath = os.sep.join((self.hyperparameters['output_dir'], self.hyperparameters['deepgru_name'] + ".pt"))
        self.resnet_fwhmx_outpath = os.sep.join((self.hyperparameters['output_dir'], self.hyperparameters['resnet_fwhmx_name'] + ".pt"))
        self.resnet_fwhmy_outpath = os.sep.join((self.hyperparameters['output_dir'], self.hyperparameters['resnet_fwhmy_name'] + ".pt"))
        self.resnet_pa_outpath = os.sep.join((self.hyperparameters['output_dir'], self.hyperparameters['resnet_pa_name'] + ".pt"))
        self.resnet_flux_outpath = os.sep.join((self.hyperparameters['output_dir'], self.hyperparameters['resnet_flux_name'] + ".pt"))
    
    
    def make(self, config):
        if config['model'] == 'blobsfinder':
            if config['mode'] == 'train':
                train_dataset = ld.ALMADataset('train_params.csv', self.train_dir, transform=self.train_compose)
                valid_dataset = ld.ALMADataset('valid_params.csv', self.valid_dir, transform=self.train_compose)
        
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=train_dataset.collate_fn)
                valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=valid_dataset.collate_fn)
            else:
                print("Preparing Data for Blobs Finder Testing....")
            test_dataset = ld.ALMADataset('test_params.csv', self.test_dir, transform=self.test_compose)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=test_dataset.collate_fn)

        else:
            if config['mode'] == 'train':
                traindata = ld.PipelineDataLoader('train_params.csv', self.train_dir)
                validdata = ld.PipelineDataLoader('valid_params.csv', self.valid_dir)
                t_spectra, t_dspectra, t_focused, t_targets, t_line_images = traindata.create_dataset()
                v_spectra, v_dspectra, v_focused, v_targets, v_line_images = validdata.create_dataset()  
            testddata = ld.PipelineDataLoader('test_params.csv', self.test_dir)
            te_spectra, te_dspectra, te_focused, te_targets, te_line_images = testddata.create_dataset()
            if config['model'] == 'spectral':
                if config['mode'] == 'train':
                    print('Preparing Data for Deep GRU Training and Testing...')
                    train_dataset = TensorDataset(torch.Tensor(t_dspectra), torch.Tensor(t_spectra))
                    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                                  pin_memory=True, shuffle=True)
                    valid_dataset = TensorDataset(torch.Tensor(v_dspectra), torch.Tensor(v_spectra))
                    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                                  pin_memory=True, shuffle=True)
                else:
                    print("Preparing Data for Deep GRU Testing....")
                test_dataset = TensorDataset(torch.Tensor(te_dspectra), torch.Tensor(te_spectra))
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(),
                                  pin_memory=True, shuffle=False)

            if config['model'] == 'resnet':
                if config['mode'] == 'train':
                    print('Preparing Data for ResNet Training and Testing...')
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
                    print("Preparing Data for ResNet Testing...")
                if config['param'] == 'flux':
                    test_dataset = TensorDataset(torch.Tensor(te_line_images), torch.Tensor(te_targets))
                else:
                    test_dataset = TensorDataset(torch.Tensor(te_focused), torch.Tensor(te_targets))
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(),
                                  pin_memory=True, shuffle=False)
        if config['mode'] == 'train':
            return train_loader, valid_loader, test_loader
        else:
            return test_loader

    def train_model(self, model, train_loader, valid_loader, config):
        example_ct = 0
        best_loss = 9999
        # initialize the early_stopping object
        if config['early_stopping']:
            early_stopping = EarlyStopping(patience=config['patience'], verbose=False)
        for epoch in tqdm(range(config.epochs)):
            running_loss = 0.0
            model.train()
            for i_batch, batch in tqdm(enumerate(train_loader)):
                inputs = batch[0].to(self.device)
                targets = batch[1]
                if config['model'] == 'resnet':
                    targets = param_selector(targets, config['param']).to(self.device)
                targets.to(self.device)
                if config['model'] == 'spectral':
                    inputs = normalize_spectra(inputs)
                    targets = normalize_spectra(targets)
                if config['model'] == 'blobsfinder':
                    loss, outputs = train_batch(inputs, targets, model, 
                                    self.blobsfinder_optimizer, self.blobsfinder_criterion)
                    train_log(loss, self.blobsfinder_optimizer, epoch)
                if config['model'] == 'spectral':
                    self.deepgru.train()
                    loss, outputs = train_batch(inputs, targets, model, 
                                    self.deepgru_optimizer, self.deepgru_criterion)
                    train_log(loss, self.deepgru_optimizer, epoch)
                if config['model'] == 'resnet':
                    self.resnet.train()
                    loss, outputs = train_batch(inputs, targets, model, 
                                    self.resnet_optimizer, self.resnet_criterion)
                    train_log(loss, self.resnet_optimizer, epoch)
                example_ct += len(inputs)
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
                    inputs = batch[0].to(self.device)
                    targets = batch[1]
                    if config['model'] == 'resnet':
                        targets = param_selector(targets, config['param']).to(self.device)
                    targets.to(self.device)
                    if config['model'] == 'spectral':
                        inputs = normalize_spectra(inputs)
                        targets = normalize_spectra(targets)
                    if config['model'] == 'blobsfinder':
                        loss, outputs = valid_batch(inputs, targets, model,
                                         self.blobsfinder_optimizer, self.blobsfinder_criterion)
                        valid_log(loss)
                        log_images(inputs, outputs, targets, 'Train')
                    if config['model'] == 'spectral':
                        loss, outputs = valid_batch(inputs, targets, model,
                                         self.deepgru_optimizer, self.deepgru_criterion)
                        valid_log(loss)
                        log_spectra(inputs, outputs, targets, 'Train')
                    if config['model'] == 'resnet':
                        loss, outputs = valid_batch(inputs, targets, model,
                                         self.resnet_optimizer, self.resnet_criterion)
                        valid_log(loss)
                        log_parameters(outputs, targets, 'Train', config)
                    running_loss += loss.item() * inputs.size(0)
                    valid_losses.append(loss.item())
            valid_loss = np.average(valid_losses)
            epoch_loss = running_loss / len(valid_loader.dataset)
            print(f"Validation Loss {epoch_loss}")
            if config['early_stopping']:
                if config['model'] == 'blobsfinder':
                    early_stopping(valid_loss, model, self.blobsfinder_optimizer, 
                            self.blobsfinder_outpath, epoch)
                if config['model'] == 'spectral':
                    early_stopping(valid_loss, model, self.deepgru_optimizer, 
                            self.deepgru_outpath, epoch)
                if config['model'] == 'resnet':
                    if config['param'] == 'fwhm_x':
                        early_stopping(valid_loss, model, self.resnet_optimizer, 
                            self.resnet_fwhmx_outpath, epoch)
                    if config['param'] == 'fwhm_y':
                        early_stopping(valid_loss, model, self.resnet_optimizer, 
                            self.resnet_fwhmy_outpath, epoch)
                    if config['param'] == 'pa':
                        early_stopping(valid_loss, model, self.resnet_optimizer, 
                            self.resnet_pa_outpath, epoch)
                    if config['param'] == 'flux':
                        early_stopping(valid_loss, model, self.resnet_optimizer, 
                            self.resnet_flux_outpath, epoch)

            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if config['model'] == 'blobsfinder':
                        save_checkpoint(model, self.blobsfinder_optimizer, self.blobsfinder_outpath, epoch)
                    if config['model'] == 'spectral':
                        save_checkpoint(model, self.deepgru_optimizer, self.deepgru_outpath, epoch)
                    if config['model'] == 'resnet':
                        if config['param'] == 'fwhm_x':
                            save_checkpoint(model, self.resnet_optimizer, self.resnet_fwhmx_outpath, epoch)
                        if config['param'] == 'fwhm_y':
                            save_checkpoint(model, self.resnet_optimizer, self.resnet_fwhmy_outpath, epoch)
                        if config['param'] == 'pa':
                            save_checkpoint(model, self.resnet_optimizer, self.resnet_pa_outpath, epoch)
                        if config['param'] == 'flux':
                            save_checkpoint(model, self.resnet_optimizer, self.resnet_flux_outpath, epoch)    
            return model

    def test_model(self, model, test_loader, config):
        tgs = []
        pds = []
        true_x = []
        true_y = []
        predicted_x = []
        predicted_y = []
        true_z = []
        predicted_z = []
        true_extension = []
        predicted_extension = []
        model.eval()
        for i_batch, batch in tqdm(enumerate(test_loader)):
            inputs = batch[0].to(self.device)
            targets = batch[1]
            if config['model'] == 'resnet':
                targets = param_selector(targets, config['param']).to(self.device)
            targets.to(self.device)
            if config['model'] == 'spectral':
                inputs = normalize_spectra(inputs)
                targets = normalize_spectra(targets)
            if config['model'] == 'blobsfinder':
                target_boxes =  batch[2]
            targets = targets.to(self.device)
            if config['model'] == 'blobsfinder':
                loss, outputs = test_batch(inputs, targets, model, self.blobsfinder_criterion)
                test_log(loss)
                for b in tqdm(range(len(targets))):
                    output = outputs[b, 0].cpu().detach().numpy()
                    min_, max_ = np.min(output), np.max(output)
                    output = (output - min_) / (max_ - min_)
                    tboxes = target_boxes[b]
                    seg = output.copy()
                    seg[seg > 0.15] = 1
                    seg = seg.astype(int)
                    struct = ndimage.generate_binary_structure(2, 2)
                    seg = binary_dilation(seg, struct)
                    props = regionprops(label(seg, connectivity=2))
                    boxes = []
                    for prop in props:
                        y0, x0, y1, x1 = prop.bbox
                        boxes.append([y0, x0, y1, x1])
                    boxes = np.array(boxes)
                    ious = box_iou(torch.Tensor(tboxes), torch.Tensor(boxes)).numpy()
                    ious = np.max(ious, axis=1)
                    txc = tboxes[:, 1] + 0.5 * (tboxes[:, 3] - tboxes[:, 1])
                    tyc = tboxes[:, 0] + 0.5 * (tboxes[:, 2] - tboxes[:, 0])
                    xc = boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1])
                    yc = boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0])
                    dists = []
                    #dists = [[np.sqrt((txc[j] - xc[k])**2 + (tyc[j] - yc[k])**2 for k in range(len(xc))] for j in range(len(txc))]
                    for j in range(len(txc)):
                        d = []
                        for k in range(len(xc)):
                            d.append(np.sqrt((txc[j] - xc[k])**2 + (tyc[j] - yc[k])**2))
                        dists.append(d)
                    dists = np.array(dists)
                    idxs = np.argmin(dists, axis=1)
                    dists = np.min(dists, axis=1)

                    for i in range(len(dists)):
                        if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                            true_x.append(txc[i])
                            true_y.append(tyc[i])
                            predicted_x.append(xc[idxs[i]])
                            predicted_y.append(yc[idxs[i]])
                            t += 1
                    if len(boxes) > len(tboxes):
                        fp += len(boxes) - len(tboxes)
            if config['model'] == 'spectral':
                loss, outputs = test_batch(inputs, targets, model, self.deepgru_criterion)
                test_log(loss)
                for b in range(len(outputs)):
                    tspectrum = targets[b, 3:127, 0].cpu().detach().numpy()
                    pspectrum = targets[b, 3:127, 0].cpu().detach().numpy()
                    min_, max_ = np.min(pspectrum), np.max(pspectrum)
                    tmin_, tmax_ = np.min(tspectrum), np.max(tspectrum)
                    y = (pspectrum - min_) / (max_ - min_)
                    ty = (tspectrum -tmin_) / (tmax_ - tmin_)
                    x = np.array(range(len(y)))
                    peaks, _ = find_peaks(y, height=np.mean(y) + 0.1, prominence=0.05, distance=10)
                    peaks_amp = y[peaks]
                    x_peaks = x[peaks]
                    tpeaks, _ = find_peaks(ty, height=0.0, prominence=0.05, distance=10)
                    tpeaks_amp = ty[tpeaks]
                    tx_peaks = x[tpeaks]
                    lims = []
                    idxs = []
                    for i_peak, peak in enumerate(x_peaks):
                        g1 = models.Gaussian1D(amplitude=peaks_amp[i_peak], mean=peak, stddev=3)
                        fit_g = fitting.LevMarLSQFitter()
                        if peak >= 10 and peak <= 118:
                            g = fit_g(g1, x[peak - 10: peak + 10], y[peak - 10: peak + 10])
                        elif peak < 10:
                            g = fit_g(g1, x[0: peak + 10], y[0: peak + 10])
                        else:
                            g = fit_g(g1, x[peak - 10: peak + 128 - peak], y[peak - 10: peak + 128 - peak])
                        m, dm = int(g.mean.value), int(g.fwhm)
                        if dm <= 64:        
                            lims.append([int(x_peaks[i_peak]) - dm, int(x_peaks[i_peak]) + dm])
                            idxs.append(i_peak)
                    tlims = []
                    tidxs = []
                    for i_peak, peak in enumerate(tx_peaks):
                        g1 = models.Gaussian1D(amplitude=tpeaks_amp[i_peak], mean=peak, stddev=3)
                        fit_g = fitting.LevMarLSQFitter()
                        if peak >= 10 and peak <= 118:
                            g = fit_g(g1, x[peak - 10: peak + 10], y[peak - 10: peak + 10])
                        elif peak < 10:
                            g = fit_g(g1, x[0: peak + 10], y[0: peak + 10])
                        else:
                            g = fit_g(g1, x[peak - 10: peak + 128 - peak], y[peak - 10: peak + 128 - peak])
                        m, dm = int(g.mean.value), int(g.fwhm)
                        if dm <= 64 and int(tx_peaks[i_peak]) - dm >= 0 and int(tx_peaks[i_peak]) + dm <= 128:
                            tlims.append([int(tx_peaks[i_peak]) - dm, int(tx_peaks[i_peak]) + dm])
                            tidxs.append(i_peak)
                    if len(lims) > 0:
                        x_peaks = x_peaks[idxs]
                        peaks_amp = peaks_amp[idxs]     
                    if len(tlims) > 0:
                        tx_peaks = tx_peaks[tidxs]
                        tpeaks_amp = tpeaks_amp[tidxs]
                        ious = []

                    if len(tlims) > 0 and len(lims) > 0:
                        dists = []
                        for j in range(len(tx_peaks)):
                            d = []
                            for k in range(len(x_peaks)):
                                d.append(np.sqrt((tx_peaks[j] - x_peaks[k]) ** 2))
                            dists.append(d)
                        dists = np.array(dists)
                        min_idxs = np.argmin(dists, axis=1)
                        dists = np.min(dists, axis=1)
                        for it, tlim in enumerate(tlims):
                            idx = min_idxs[it]
                            ious.append(iou(tlims[it], lims[idx]))
                        ious = np.array(ious)
                        for i in range(len(dists)):
                            if ious[i] >= config['oneD_iou_threshold'] and dists[i] <= config['oneD_dist_threshold']:
                                t += 1
                                true_z.append(tx_peaks[i])
                                predicted_z.append(x_peaks[min_idxs[i]])
                                true_extension.append(tlims[i][1] - tlims[i][0])
                                predicted_extension.append(lims[min_idxs[i]][1] -lims[min_idxs[i]][0])

                        if len(x_peaks) > len(tx_peaks):
                            fp += len(x_peaks) - len(tx_peaks)
            
            if config['model'] == 'resnet':
                loss, outputs = test_batch(inputs, targets, model, self.resnet_criterion)
                test_log(loss)
                target = targets[b].cpu().detach().numpy()
                prediction = outputs[b].cpu().detach().numpy()
                for i in range(len(target)):
                    tgs.append(target[i])
                    pds.append(prediction[i])
        tgs = np.array(tgs)
        pds = np.array(pds)
        res = tgs - pds
        if config['model'] == 'blobsfinder':
            true_x = np.array(true_x)
            true_y = np.array(true_y)
            predicted_x = np.array(predicted_x)
            predicted_y = np.array(predicted_y)
            return t, len(test_loader.dataset), fp, true_x, true_y, predicted_x, predicted_y
        if config['model'] == 'spectral':
            true_z = np.array(true_z)
            predicted_z = np.array(predicted_z)
            true_extension = np.array(true_extension)
            predicted_extension = np.array(predicted_extension)
            return t, len(test_loader.dataset), fp, true_z, true_extension, predicted_z, predicted_extension
        if config['model'] == 'resnet':
            return tgs, pds, res

    def train_and_test_model(self):
        if self.hyperparameters['mode'] == 'train':
            with wandb.init(project=self.hyperparameters['project'],
                             name=self.hyperparameters['name'], 
                             entity=self.hyperparameters['entity'], 
                             config=self.hyperparameters):
                config = wandb.config
                
                train_loader, valid_loader, test_loader = self.make(config)
                print('Training...')
                if config['model'] == 'boobsfinder':
                    self.train_model(self.blobsfinder, train_loader, valid_loader, config)
                if config['model'] == 'spectral':
                    self.train_model(self.deepgru, train_loader, valid_loader, config)
                if config['model'] == 'resnet':
                    self.train_model(self.resnet, train_loader, valid_loader, config)
                print('Testing...')
                if config['model'] == 'resnet':
                    tgs, pds, res = self.test_model(self.resnet, test_loader, config)
                    truth_name = config['prediction_dir'] + "/" + config['param'] + '_' + config['name'] + '_targets.npy'
                    prediction_name = config['prediction_dir'] + "/" + config['param'] + '_' + config['name'] + '_predictions.npy'
                    residual_name = config['prediction_dir'] + "/" + config['param'] + '_' + config['name'] + '_residuals.npy'
                    np.save(truth_name, tgs)
                    np.save(prediction_name, pds)
                    np.save(residual_name, res)

                    fig = plt.figure(figsize=(8, 8))
                    plt.hist(res, bins=50, edgecolor='black', color='dodgerblue')
                    plt.xlabel(config['param'] + ' residuals')
                    plt.ylabel('N')
                    outhname = config['plot_dir'] + '/' + config['param'] + '_residuals.png'
                    plt.savefig(outhname)
                    wandb.log({"Residuals": fig})

                    plt.figure(figsize=(8, 8))
                    plt.scatter(tgs, pds, s=1, c='dodgerblue')
                    y_lim = plt.ylim()
                    x_lim = plt.xlim()
                    plt.plot(x_lim, y_lim, color = 'r', linestyle='dashed')
                    plt.xlabel('True ' + config['param'] )
                    plt.ylabel('Prdicted ' + config['param'])
                    outhname = config['plot_dir'] + '/' + config['param'] + '_scatter.png'
                    plt.savefig(outhname)
                    wandb.log({"Scatter": fig})
                    return tgs, pds, res
                if config['model'] == 'blobsfinder':
                    tp, tot, fp, true_x, true_y, predicted_x, predicted_y = self.test_model(self.blobsfinder, test_loader, config)
                    truth_x_name = config['prediction_dir'] + "/x_" + config['name'] + '_targets.npy'
                    truth_y_name = config['prediction_dir'] + "/y_"  + config['name'] + '_targets.npy'
                    prediction_x_name = config['prediction_dir'] + "/x_" + config['name'] + '_prediction.npy'
                    prediction_y_name = config['prediction_dir'] + "/y_"  + config['name'] + '_prediction.npy'
                    np.save(truth_x_name, true_x)
                    np.save(truth_y_name, true_y)
                    np.save(prediction_x_name, predicted_x)
                    np.save(prediction_y_name, predicted_y)
                    return tp, tot, fp
                if config['model'] == 'spectral':
                    tp, tot, fp, true_z, true_extension, predicted_z, predicted_extension = self.test_model(self.deepgru, test_loader, config)
                    truth_z_name = config['prediction_dir'] + "/z_" + config['name'] + '_targets.npy'
                    truth_extension_name = config['prediction_dir'] + "/z_extension_"  + config['name'] + '_targets.npy'
                    prediction_z_name = config['prediction_dir'] + "/z_" + config['name'] + '_prediction.npy'
                    prediction_extension_name = config['prediction_dir'] + "/z_extension_"  + config['name'] + '_prediction.npy'
                    np.save(truth_z_name, true_z)
                    np.save(truth_extension_name, true_extension)
                    np.save(prediction_z_name, predicted_z)
                    np.save(prediction_extension_name, predicted_extension)
                    return tp, tot, fp
        else:
            with wandb.init(project=self.hyperparameters['project'],
                             name=self.hyperparameters['name'] + '_Test', 
                             entity=self.hyperparameters['entity'],
                             config=self.hyperparameters, 
                             ):
                config = wandb.config
                test_loader = make(config, self.device)
                print('Testing...')
                if config['model'] == 'resnet':
                    tgs, pds, res = self.test_model(self.resnet, test_loader, config)
                    truth_name = config['prediction_dir'] + "/" + config['param'] + '_' + config['name'] + '_targets.npy'
                    prediction_name = config['prediction_dir'] + "/" + config['param'] + '_' + config['name'] + '_predictions.npy'
                    residual_name = config['prediction_dir'] + "/" + config['param'] + '_' + config['name'] + '_residuals.npy'
                    np.save(truth_name, tgs)
                    np.save(prediction_name, pds)
                    np.save(residual_name, res)


                    fig = plt.figure(figsize=(8, 8))
                    plt.hist(res, bins=50, edgecolor='black', color='dodgerblue')
                    plt.xlabel(config['param'] + ' residuals')
                    plt.ylabel('N')
                    outhname = config['plot_dir'] + '/' + config['param'] + '_residuals.png'
                    plt.savefig(outhname)
                    wandb.log({"Residuals": fig})

                    plt.figure(figsize=(8, 8))
                    plt.scatter(tgs, pds, s=1, c='dodgerblue')
                    y_lim = plt.ylim()
                    x_lim = plt.xlim()
                    plt.plot(x_lim, y_lim, color = 'r', linestyle='dashed')
                    plt.xlabel('True ' + config['param'] )
                    plt.ylabel('Prdicted ' + config['param'])
                    outhname = config['plot_dir'] + '/' + config['param'] + '_scatter.png'
                    plt.savefig(outhname)
                    wandb.log({"Scatter": fig})
                if config['model'] == 'blobsfinder':
                    tp, tot, fp, true_x, true_y, predicted_x, predicted_y = self.test_model(self.blobsfinder, test_loader, config)
                    truth_x_name = config['prediction_dir'] + "/x_" + config['name'] + '_targets.npy'
                    truth_y_name = config['prediction_dir'] + "/y_"  + config['name'] + '_targets.npy'
                    prediction_x_name = config['prediction_dir'] + "/x_" + config['name'] + '_predictions.npy'
                    prediction_y_name = config['prediction_dir'] + "/y_"  + config['name'] + '_predictions.npy'
                    np.save(truth_x_name, true_x)
                    np.save(truth_y_name, true_y)
                    np.save(prediction_x_name, predicted_x)
                    np.save(prediction_y_name, predicted_y)
                    wandb.log({'Detections': tp, 'Total': tot, 'FP': fp })
                if config['model'] == 'spectral':
                    tp, tot, fp, true_z, true_extension, predicted_z, predicted_extension = self.test_model(self.deepgru, test_loader, config)
                    truth_z_name = config['prediction_dir'] + "/z_" + config['name'] + '_targets.npy'
                    truth_extension_name = config['prediction_dir'] + "/z_extension_"  + config['name'] + '_targets.npy'
                    prediction_z_name = config['prediction_dir'] + "/z_" + config['name'] + '_predictions.npy'
                    prediction_extension_name = config['prediction_dir'] + "/z_extension_"  + config['name'] + '_predictions.npy'
                    np.save(truth_z_name, true_z)
                    np.save(truth_extension_name, true_extension)
                    np.save(prediction_z_name, predicted_z)
                    np.save(prediction_extension_name, predicted_extension)
                    wandb.log({'Detections': tp, 'Total': tot, 'FP': fp })

    def select_criterion(self, criterion_name):
        if isinstance(criterion_name, list):
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
        return criterion

    def get_spectral_loader_from_blobsfinder_predictions(self, model, dataset, loader, criterion, config, test=True):
        model.eval()
        dirty_list = dataset.dirty_list
        clean_list = dataset.clean_list
        parameters = dataset.parameters
        targs = []
        true_x = []
        true_y = []
        predicted_x = []
        predicted_y = []
        spectra = []
        dspectra = []
        t = 0
        fp = 0
        for i_batch, batch in tqdm(enumerate(loader)):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            target_boxes =  batch[2]
            targets = targets.to(self.device)
            idxs = batch[3]
            loss, outputs = test_batch(inputs, targets, model, criterion)
            for b in tqdm(range(len(outputs))):
                output = outputs[b, 0].cpu().detach().numpy()
                dirty_name = dirty_list[idxs[b]]
                clean_name = clean_list[idxs[b]]
                db_idx = float(dirty_name.split('_')[-1].split('.')[0])
                params = parameters.loc[parameters.ID == db_idx]
                boxes = np.array(params[["y0", "x0", "y1", "x1"]].values)
                z_ = np.array(params["z"].values)
                fwhm_z = np.array(params["fwhm_z"].values)
                extensions = 2 * fwhm_z 
                targets = np.array(params[["x", "y", "fwhm_x", "fwhm_y", "pa", "flux", 'continuum']].values)
                targets['z'] = z_
                targets['extensions'] = extensions
                dirty_cube = fits.getdata(dirty_name) 
                clean_cube = fits.getdata(clean_name)
                min_, max_ = np.min(output), np.max(output)
                output = (output - min_) / (max_ - min_)
                tboxes = target_boxes[b]
                seg = output.copy()
                seg[seg > 0.15] = 1
                seg = seg.astype(int)
                struct = ndimage.generate_binary_structure(2, 2)
                seg = binary_dilation(seg, struct)
                props = regionprops(label(seg, connectivity=2))
                boxes = []
                for prop in props:
                    y0, x0, y1, x1 = prop.bbox
                    boxes.append([y0, x0, y1, x1])
                boxes = np.array(boxes)
                dirty_spectra = np.array([
                    np.sum(dirty_cube[0][:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                    for j in range(len(boxes))
                    ])
                clean_spectra = np.array([
                    np.sum(clean_cube[0][:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                    for j in range(len(boxes))
                    ])
                for j in range(len(dirty_spectra)):
                    targs.append(targets[j])
                    dspec = dirty_spectra[j]
                    dspec = (dspec - np.mean(dspec)) / np.std(dspec)
                    spec = clean_spectra[j]
                    spec = (spec - np.mean(spec)) / np.std(spec)
                    spectra.append(spec)
                    dspectra.append(dspec)
                ious = box_iou(torch.Tensor(tboxes), torch.Tensor(boxes)).numpy()
                ious = np.max(ious, axis=1)
                txc = tboxes[:, 1] + 0.5 * (tboxes[:, 3] - tboxes[:, 1])
                tyc = tboxes[:, 0] + 0.5 * (tboxes[:, 2] - tboxes[:, 0])
                xc = boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1])
                yc = boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0])
                dists = []
                for j in range(len(txc)):
                    d = []
                    for k in range(len(xc)):
                        d.append(np.sqrt((txc[j] - xc[k])**2 + (tyc[j] - yc[k])**2))
                    dists.append(d)
                dists = np.array(dists)
                d_idxs = np.argmin(dists, axis=1)
                dists = np.min(dists, axis=1)
                for i in range(len(dists)):
                    if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                        true_x.append(txc[i])
                        true_y.append(tyc[i])
                        predicted_x.append(xc[d_idxs[i]])
                        predicted_y.append(yc[d_idxs[i]])
                        t += 1
                if len(boxes) > len(tboxes):
                    fp += len(boxes) - len(tboxes)
                
        true_x = np.array(true_x)
        true_y = np.array(true_y)
        predicted_x = np.array(predicted_x)
        predicted_y = np.array(predicted_y)
        spectra = np.array(spectra)
        dspectra = np.array(dspectra)
        targs = np.array(targs)
        dspectra = np.transpose(dspectra[np.newaxis], 
                                        axes=(1, 2, 0))
        spectra = np.transpose(spectra[np.newaxis], 
                                        axes=(1, 2, 0)) 
        dataset = TensorDataset(torch.Tensor(dspectra), torch.Tensor(spectra), torch.Tensor(targs))
        if test:
            loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                                  pin_memory=True, shuffle=False)
        else:  
            loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=os.cpu_count(), 
                                  pin_memory=True, shuffle=True)
        return loader

    def train_and_test_pipeline(self):
        
        print('Loading Checkpoints for all models')
        
        blobsfinder, _, _ = load_checkpoint(self.blobsfinder, 
                                self.blobsfinder_optimizer, 
                                self.blobsfinder_outpath)
        deepgru, _, _ = load_checkpoint(self.deepgru, 
                                self.deepgru_optimizer, 
                                self.deepgru_outpath)
        resnet_fwhmx, _, _ = load_checkpoint(self.resnet, 
                                self.resnet_optimizer, 
                                self.resnet_fwhmx_outpath)
        resnet_fwhmy, _, _ = load_checkpoint(self.resnet, 
                                self.resnet_optimizer, 
                                self.resnet_fwhmy_outpath)
        resnet_pa, _, _ = load_checkpoint(self.resnet, 
                                self.resnet_optimizer, 
                                self.resnet_pa_outpath)
        resnet_flux, _, _ = load_checkpoint(self.resnet, 
                                self.resnet_optimizer, 
                                self.resnet_flux_outpath)
        train_dataset = ld.ALMADataset('train_params.csv', self.train_dir, 
                                                transform=self.test_compose)
        valid_dataset = ld.ALMADataset('valid_params.csv', self.valid_dir, 
                                                transform=self.test_compose)
        train_loader = DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=train_dataset.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=self.hyperparameters['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=valid_dataset.collate_fn)
        test_dataset = ld.ALMADataset('test_params.csv', self.test_dir, transform=self.test_compose)
        test_loader = DataLoader(test_dataset, batch_size=self.hyperparameters['batch_size'], num_workers=os.cpu_count(), 
                              pin_memory=True, shuffle=True, collate_fn=test_dataset.collate_fn)
        if self.hyperparameters['mode'] == 'train':
            with wandb.init(project=self.hyperparameters['project'],
                             name=self.hyperparameters['name'], 
                             entity=self.hyperparameters['entity'], 
                             config=self.hyperparameters):
                config = wandb.config
                deepgru_train_loader = self.get_spectral_loader_from_blobsfinder_predictions(
                    blobsfinder, train_dataset, train_loader, self.blobsfinder_criterion, config, False
                )
                deepgru_valid_loader = self.get_spectral_loader_from_blobsfinder_predictions(
                    blobsfinder, valid_dataset, valid_loader, self.blobsfinder_criterion, config, False
                )
                deepgru_test_loader = self.get_spectral_loader_from_blobsfinder_predictions(
                    blobsfinder, test_dataset, test_loader, self.blobsfinder_criterion, config, True
                )
                self.train_model(deepgru, deepgru_train_loader, deepgru_valid_loader, config)




                

                
        
                

        else:
            with wandb.init(project=self.hyperparameters['project'],
                             name=self.hyperparameters['name'] + '_Test', 
                             entity=self.hyperparameters['entity'],
                             config=self.hyperparameters, 
                             ):
                config = wandb.config


            
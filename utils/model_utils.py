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

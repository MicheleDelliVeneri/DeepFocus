import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from torchvision import transforms
from torchmetrics import MeanSquaredLogError
import numpy as np
import pandas as pd
import models.blobsfinder as bf
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
matplotlib.rcParams.update({'font.size': 12})
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("setting random seeds") % 2**32 - 1)
torch.manual_seed(hash("setting random seeds") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("setting random seeds") % 2**32 - 1)

# ------------------------- CLASSES AND FUNCTIONS --------------------------------------
class DECORAS_Enc2D(nn.Module):
    def __init__(self):
        super(DECORAS_Enc2D, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        self.dense = nn.Linear(64 * 16 * 16, 256)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x

class DECORAS_Dec2D(nn.Module):
    def __init__(self):
        super(DECORAS_Dec2D, self).__init__()

        self.dense = nn.Linear(256,  64  * 16 * 16)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 16, 16))
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), bias=False, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), bias=False, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16))
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), bias=False, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8))
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), bias=False, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())
        self.out = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(1, 1), bias=False, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.dense(x)
        #print(x.size())
        x = self.unflatten(x)
        #print(x.size())
        x = self.block1(x)
        #print(x.size())
        x = self.block2(x)
        #print(x.size())
        x = self.block3(x)
        #print(x.size())
        x = self.block4(x)
        #print(x.size())
        x = self.out(x)
        #print(x.size())
        return x
            
class DECORAS_BF(nn.Module):
    def __init__(self):
        super(DECORAS_BF, self).__init__()
    
        self.enc = DECORAS_Enc2D()
        self.dec = DECORAS_Dec2D()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        lat = self.enc(x)
        out = self.dec(lat)
        return out

class DECORAS_Enc2DSS(nn.Module):
    def __init__(self):
        super(DECORAS_Enc2DSS, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), bias=False, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.dense = nn.Linear(32 * 16 * 16, 64)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x

class DECORAS_Dec2DSS(nn.Module):
    def __init__(self):
        super(DECORAS_Dec2DSS, self).__init__()

        self.dense = nn.Linear(64,  32  * 16 * 16)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 16, 16))
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), bias=False, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16))
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), bias=False, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8))
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), bias=False, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU())
        self.out = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(1, 1), bias=False, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.dense(x)
        #print(x.size())
        x = self.unflatten(x)
        #print(x.size())
        x = self.block1(x)
        #print(x.size())
        x = self.block2(x)
        #print(x.size())
        x = self.block3(x)
        #print(x.size())
        x = self.out(x)
        #print(x.size())
        return x

class DECORAS_BF2(nn.Module):
    def __init__(self):
        super(DECORAS_BF2, self).__init__()
    
        self.enc = DECORAS_Enc2DSS()
        self.dec = DECORAS_Dec2DSS()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        lat = self.enc(x)
        out = self.dec(lat)
        return out

class batch_act(nn.Module):
    def __init__(self, in_c):
        super(batch_act, self).__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.act = nn.LeakyReLU()
    def forward(self, inputs):
        x = self.act(inputs)
        x = self.bn(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(residual_block, self).__init__()
        # Convolutional Layers
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b1 = batch_act(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.b2 = batch_act(out_c)

        #shortcut connection 
        #self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)
    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.c2(x)
        x = self.b2(x)
        #s = self.s(inputs)
        #x = x + s
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        self.c = nn.ConvTranspose2d(in_c, out_c, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.r = residual_block(in_c + out_c, out_c)
    
    def forward(self, inputs):
        x = self.upsample(inputs)
        x1 = self.c(inputs)
        x = torch.cat([x, x1], dim=1)
        #print(x.shape, skip.shape)
        x = self.r(x)
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_c):
        super(Encoder, self).__init__()
        self.b1 = residual_block(1, 8, stride=2)
        self.b2 = residual_block(8, 16, stride=2)
        self.b3 = residual_block(16, 32, stride=2)
        self.b4 = residual_block(32, 64, stride=2)
        self.dense = nn.Sequential(
            nn.Linear(64 * 16 *16, hidden_c),
            nn.LeakyReLU())
    
    def forward(self, inputs):
        #print(inputs.shape)
        skip1 = self.b1(inputs)
        #print(skip1.shape)
        skip2 = self.b2(skip1)
        #print(skip2.shape)
        skip3 = self.b3(skip2)
        #print(skip3.shape)
        x = self.b4(skip3)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = self.dense(x)
        #print(x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_c):
        super(Decoder, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_c, 64 * 16 * 16),
            nn.LeakyReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 16, 16))
        self.b1 = decoder_block(64, 32)
        self.b2 = decoder_block(32, 16)
        self.b3 = decoder_block(16, 8)
        self.b4 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True),
            residual_block(8, 1))
        self.out = nn.Sequential(
            nn.Conv2d(1,  1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        #print(inputs.shape)
        x = self.dense(inputs)
        #print(x.shape)
        x = self.unflatten(x)
        #print(x.shape, skip3.shape)
        x = self.b1(x)
        #print(x.shape, skip2.shape)
        x = self.b2(x)
        #print(x.shape, skip1.shape)
        x = self.b3(x)
        #print(x.shape)
        x = self.b4(x)
        #print(x.shape)
        x = self.out(x)
        #print(x.shape)
        return x

class BlobsFinder(nn.Module):
    def __init__(self, hidden_c):
        super(BlobsFinder, self).__init__()
        self.enc = Encoder(hidden_c)
        self.dec = Decoder(hidden_c)
        self.__init__params()
        
    def __init__params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,   0)

    def forward(self, inputs):
        lat = self.enc(inputs)
        out = self.dec(lat)
        return out

def make_blobsfinder(config, device):
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
                          pin_memory=True, shuffle=True, collate_fn=test_dataset.collate_fn)
    model = BlobsFinder(config['hidden_channels'])
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

def blobsfinder_train_model(hyperparameters, device, name):
    with wandb.init(project='blobsfinder', name=name, entity='bradipo', config=hyperparameters):
        config = wandb.config
        blobsfinder, criterion, optimizer, train_loader, valid_loader = make_blobsfinder(config, device)
        train(blobsfinder, train_loader, valid_loader, criterion, optimizer, config, name, device)

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
    checkpoint = torch.load(load_path, map_location='cuda:0')
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

def train(model, train_loader, valid_loader, criterion, optimizer, config, name, device):
    example_ct = 0
    best_loss = 9999
    # initialize the early_stopping object
    if config['early_stopping']:
        early_stopping = EarlyStopping(patience=config['patience'], verbose=False)
    outpath = os.sep.join((config['output_dir'], name + ".pt"))
    for epoch in tqdm(range(config.epochs)):
        model.train()
        running_loss = 0.0
        for i_batch, batch in tqdm(enumerate(train_loader)):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            loss, outputs = train_batch(inputs, targets, model, optimizer, criterion)
            example_ct += len(inputs)
            train_log(loss, optimizer, epoch)
            if i_batch == len(train_loader) - 1:
                log_images(inputs, outputs, targets, 'Train')
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss {epoch_loss}")
        torch.cuda.empty_cache()
        model.eval()
        running_loss = 0.0
        valid_losses = []
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(valid_loader)):
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                loss, outputs = valid_batch(inputs, targets, model, optimizer, criterion)
                valid_log(loss)
                if i_batch == len(valid_loader) - 1:
                    log_images(inputs, outputs, targets, 'Validation')
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

def test_blobsfinder(model, test_loader, criterion, config, device):
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    if not os.path.exists(config['plot_dir']):
        os.mkdir(config['plot_dir'])
    if not os.path.exists(config['prediction_dir']):
        os.mkdir(config['prediction_dir'])
    true_x = []
    true_y = []
    predicted_x = []
    predicted_y = []
    IoUs = []
    dIoUs = []
    tIoUs = []
    dfluxes = []
    tfluxes = []
    pSNRs = []
    tSNRs = []
    for i_batch, batch in tqdm(enumerate(test_loader)):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        target_boxes =  batch[2]
        snrs = batch[3]
        target_parameters = batch[4]
        loss, outputs = test_batch(inputs, targets, model, criterion)
        for b in tqdm(range(len(targets))):
            tboxes = target_boxes[b]
            tsnrs = snrs[b]
            fluxes = target_parameters[b][:, 5]
            tboxes_ious = np.max(remove_diag(box_iou(torch.Tensor(tboxes), torch.Tensor(tboxes)).numpy()), axis=1)
            output = outputs[b, 0].cpu().detach().numpy()
            min_, max_ = np.min(output), np.max(output)
            seg = output.copy()
            seg[seg > 0.15] = 1
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
                #if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                if dists[i] <= config['twoD_dist_threshold']:
                    true_x.append(txc[i])
                    true_y.append(tyc[i])
                    predicted_x.append(pxs[idxs[i]])
                    predicted_y.append(pys[idxs[i]])
                    dIoUs.append(tboxes_ious[i])
                    dfluxes.append(fluxes[i])
                    pSNRs.append(tsnrs[i])
                    tp += 1
                else:
                    fn += 1
                tIoUs.append(tboxes_ious[i])
                tSNRs.append(tsnrs[i])
                tfluxes.append(fluxes[i])
                IoUs.append(ious[i])
            if len(boxes) > len(tboxes):
                fp += len(boxes) - len(tboxes)
    pSNRs = np.array(pSNRs)
    tSNRs = np.array(tSNRs)
    true_x = np.array(true_x)
    true_y = np.array(true_y)
    predicted_x = np.array(predicted_x)
    predicted_y = np.array(predicted_y)
    IoUs = np.array(IoUs)
    mean_IoU = np.mean(IoUs)
    tIoUs = np.array(tIoUs)
    tfluxes = np.array(tfluxes)
    dfluxes = np.array(dfluxes)
    dIoUs = np.array(dIoUs)
    blobsfinder_x_predictions_name = os.path.join(config['prediction_dir'], 'blobsfinder_x_predictions.npy')
    blobsfinder_y_predictions_name = os.path.join(config['prediction_dir'], 'blobsfinder_y_predictions.npy')
    blobsfinder_x_true_name = os.path.join(config['prediction_dir'], 'blobsfinder_x_true.npy')
    blobsfinder_y_true_name = os.path.join(config['prediction_dir'], 'blobsfinder_y_true.npy')
    np.save(blobsfinder_x_predictions_name, predicted_x)
    np.save(blobsfinder_y_predictions_name, predicted_y)
    np.save(blobsfinder_x_true_name, true_x)
    np.save(blobsfinder_y_true_name, true_y)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    #names = np.array(['tp', 'fp', 'fn', 'Precision', 'Recall', 'Mean IoU'])
    #values = np.array([tp, fp, fn, precision, recall, mean_IoU])
    #rdb = pd.DataFrame(data=values, columns=names)
    #result_name = os.path.join(config['prediction_dir'], 'decoras_results.csv')
    #rdb.to_csv(result_name)
    return tp, len(test_loader.dataset), fp, fn, precision, recall, mean_IoU, true_x, true_y, predicted_x, predicted_y, dIoUs, dfluxes, tIoUs, tfluxes

def test_single_model(model, test_loader, criterion, config, device):
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    if not os.path.exists(config['plot_dir']):
        os.mkdir(config['plot_dir'])
    if not os.path.exists(config['prediction_dir']):
        os.mkdir(config['prediction_dir'])
    true_x = []
    true_y = []
    predicted_x = []
    predicted_y = []
    IoUs = []
    dIoUs = []
    tIoUs = []
    dfluxes = []
    tfluxes = []
    pSNRs = []
    tSNRs = []
    for i_batch, batch in tqdm(enumerate(test_loader)):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        target_boxes =  batch[2]
        snrs = batch[3]
        target_parameters = batch[4]
        loss, outputs = test_batch(inputs, targets, model, criterion)
        for b in tqdm(range(len(targets))):
            tboxes = target_boxes[b]
            tsnrs = snrs[b]
            tboxes_ious = np.max(remove_diag(box_iou(torch.Tensor(tboxes), torch.Tensor(tboxes)).numpy()), axis=1)
            output = outputs[b, 0].cpu().detach().numpy()
            min_, max_ = np.min(output), np.max(output)
            output = 100* (output - min_) / (max_ - min_)
            blobs = blob_dog(output)
            fluxes = target_parameters[b][:, 5]
            radiuses = np.sqrt((3 * blobs[:, 2])**2)
            pxs = blobs[:, 1]
            pys = blobs[:, 0]
            boxes = []
            for j in range(len(radiuses)):
                y0 = pys[j] - radiuses[j]
                y1 = pys[j] + radiuses[j]
                x0 = pxs[j] - radiuses[j]
                x1 = pxs[j] + radiuses[j]
                boxes.append([y0, x0, y1, x1])
            boxes = np.array(boxes)
            txc = tboxes[:, 1] + 0.5 * (tboxes[:, 3] - tboxes[:, 1])
            tyc = tboxes[:, 0] + 0.5 * (tboxes[:, 2] - tboxes[:, 0])
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
                #if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                if dists[i] <= config['twoD_dist_threshold']:
                    true_x.append(txc[i])
                    true_y.append(tyc[i])
                    predicted_x.append(pxs[idxs[i]])
                    predicted_y.append(pys[idxs[i]])
                    dIoUs.append(tboxes_ious[i])
                    dfluxes.append(fluxes[i])
                    pSNRs.append(tsnrs[i])
                    tp += 1
                else:
                    fn += 1
                tIoUs.append(tboxes_ious[i])
                tSNRs.append(tsnrs[i])
                tfluxes.append(fluxes[i])
                IoUs.append(ious[i])

            if len(boxes) > len(tboxes):
                fp += len(boxes) - len(tboxes)
    pSNRs = np.array(pSNRs)
    tSNRs = np.array(tSNRs)
    true_x = np.array(true_x)
    true_y = np.array(true_y)
    predicted_x = np.array(predicted_x)
    predicted_y = np.array(predicted_y)
    IoUs = np.array(IoUs)
    mean_IoU = np.mean(IoUs)
    tIoUs = np.array(tIoUs)
    tfluxes = np.array(tfluxes)
    dfluxes = np.array(dfluxes)
    dIoUs = np.array(dIoUs)
    decoras_x_predictions_name = os.path.join(config['prediction_dir'], 'decoras_x_predictions.npy')
    decoras_y_predictions_name = os.path.join(config['prediction_dir'], 'decoras_y_predictions.npy')
    decoras_x_true_name = os.path.join(config['prediction_dir'], 'decoras_x_true.npy')
    decoras_y_true_name = os.path.join(config['prediction_dir'], 'decoras_y_true.npy')
    np.save(decoras_x_predictions_name, predicted_x)
    np.save(decoras_y_predictions_name, predicted_y)
    np.save(decoras_x_true_name, true_x)
    np.save(decoras_y_true_name, true_y)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    #names = np.array(['tp', 'fp', 'fn', 'Precision', 'Recall', 'Mean IoU'])
    #values = np.array([tp, fp, fn, precision, recall, mean_IoU])
    #rdb = pd.DataFrame(data=values, columns=names)
    #result_name = os.path.join(config['prediction_dir'], 'decoras_results.csv')
    #rdb.to_csv(result_name)
    return tp, len(test_loader.dataset), fp, fn, precision, recall, mean_IoU, true_x, true_y, predicted_x, predicted_y, dIoUs, dfluxes, tIoUs, tfluxes

def test(model1, model2, test_loader, criterion1, criterion2, config, device):
    model1.eval()
    fp = 0
    fn = 0
    if not os.path.exists(config['plot_dir']):
        os.mkdir(config['plot_dir'])
    if not os.path.exists(config['prediction_dir']):
        os.mkdir(config['prediction_dir'])
    true_x = []
    true_y = []
    predicted_x = []
    predicted_y = []
    IoUs = []
    for i_batch, batch in tqdm(enumerate(test_loader)):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        target_boxes =  batch[2]
        loss, outputs1 = test_batch(inputs, targets, model1, criterion1)
        loss, outputs2 = test_batch(inputs, targets, model2, criterion2)
        target_parameters = batch[4]
        for b in tqdm(range(len(targets))):
            output1 = outputs1[b, 0].cpu().detach().numpy()
            min_, max_ = np.min(output1), np.max(output1)
            output1 = (output1 - min_) / (max_ - min_)
            tboxes = target_boxes[b]
            tboxes_ious = box_iou(torch.Tensor(tboxes), torch.Tensor(tboxes)).numpy()
            fluxes = target_parameters[b][:, 5]
            blobs1 = blob_dog(output1, min_sigma=3)
            pxs1 = blobs1[:, 0]
            pys1 = blobs1[:, 1]
            radiuses = np.sqrt(2 * blobs1[:, 2])
            output2 = outputs2[b, 0].cpu().detach().numpy()
            min_, max_ = np.min(output2), np.max(output2)
            output2 = (output2 - min_) / (max_ - min_)
            tboxes = target_boxes[b]
            blobs2 = blob_dog(output2, min_sigma=3)
            pxs2 = blobs2[:, 0]
            pys2 = blobs2[:, 1]
            tpxs, tpys = [], []
            idxs = []
            # Selecting only blobs which are predicted by both models
            for j in range(len(pxs1)):
                px1 = pxs1[j]
                py1 = pys1[j]
                k = 0
                for k in range(len(pxs2)):
                    dist = np.sqrt((px1 - pxs2[k])**2 + (py1 - pys2[k])**2)
                    if dist <= 3 and k < 1:
                        tpxs.append(px1)
                        tpys.append(py1)
                        idxs.append(j)
            tpxs = np.array(tpxs)
            tpys = np.array(tpys)
            idxs = np.array(idxs)
            radiuses = radiuses[idxs]
            boxes = []
            # measuing boundinb boxes from radiuses and centers
            for j in range(len(radiuses)):
                y0 = tpys[j] - radiuses[j]
                y1 = tpys[j] + radiuses[j]
                x0 = tpxs[j] - radiuses[j]
                x1 = tpxs[j] + radiuses[j]
                boxes.append([y0, x0, y1, x1])
            boxes = np.array(boxes)
            txc = tboxes[:, 1] + 0.5 * (tboxes[:, 3] - tboxes[:, 1])
            tyc = tboxes[:, 0] + 0.5 * (tboxes[:, 2] - tboxes[:, 0])
            # merasuring distances and IoUs between true and predicted bounding boxes
            dists = []
            for j in range(len(txc)):
                d = []
                for k in range(len(tpxs)):
                    d.append(np.sqrt((txc[j] - tpxs[k])**2 + (tyc[j] - tpys[k])**2))
                    dists.append(d)
            dists = np.array(dists)
            idxs = np.argmin(dists, axis=1)
            dists = np.min(dists, axis=1)
            ious = box_iou(torch.Tensor(tboxes), torch.Tensor(boxes)).numpy()
            ious = np.max(ious, axis=1)
            for i in range(len(dists)):
                if ious[i] >= config['twoD_iou_threshold'] and dists[i] <= config['twoD_dist_threshold']:
                    true_x.append(txc[i])
                    true_y.append(tyc[i])
                    predicted_x.append(tpxs[idxs[i]])
                    predicted_y.append(tpys[idxs[i]])
                    tp += 1
                else:
                    fn += 1
                IoUs.append(ious[i])
            if len(boxes) > len(tboxes):
                fp += len(boxes) - len(tboxes)
    
    true_x = np.array(true_x)
    true_y = np.array(true_y)
    predicted_x = np.array(predicted_x)
    predicted_y = np.array(predicted_y)
    IoUs = np.array(IoUs)
    mean_IoU = np.mean(IoUs)
    decoras_x_predictions_name = os.path.join(config['prediction_dir'], 'decoras_x_predictions.npy')
    decoras_y_predictions_name = os.path.join(config['prediction_dir'], 'decoras_y_predictions.npy')
    decoras_x_true_name = os.path.join(config['prediction_dir'], 'decoras_x_true.npy')
    decoras_y_true_name = os.path.join(config['prediction_dir'], 'decoras_y_true.npy')
    np.save(decoras_x_predictions_name, predicted_x)
    np.save(decoras_y_predictions_name, predicted_y)
    np.save(decoras_x_true_name, true_x)
    np.save(decoras_y_true_name, true_y)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    names = ['tp', 'fp', 'fn', 'Precision', 'Recall', 'Mean IoU']
    values = [tp, fp, fn, precision, recall, mean_IoU]
    rdb = pd.DataFrame(data=values, columns=names)
    result_name = os.path.join(config['prediction_dir'], 'decoras_results.csv')
    rdb.to_csv(result_name)
    return tp, len(test_loader.dataset), fp, fn, precision, recall, mean_IoU, true_x, true_y, predicted_x, predicted_y

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def make(config, device):
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
                          pin_memory=True, shuffle=True, collate_fn=test_dataset.collate_fn)
    model1 = DECORAS_BF()
    model2 = DECORAS_BF()
    if torch.cuda.device_count() > 1 and config['multi_gpu']:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
    print(f'Using {device}') 
    model1.to(device)
    model2.to(device)
    criterion1 = RMSLELoss()
    criterion2 = nn.BCELoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=config['learning_rate'], 
                                         weight_decay=config['weight_decay'])
    if config['mode'] == 'train':
        return model1, model2, criterion1, criterion2, optimizer1, optimizer2, train_loader, valid_loader
    else:
        print('Loading Checkpoint....')
        outpath1 = os.sep.join((config['output_dir'], config['model1_name'] + ".pt"))
        outpath2 = os.sep.join((config['output_dir'], config['model2_name'] + ".pt"))
        model1, _, _ = load_checkpoint(model1, optimizer1, outpath1)
        model2, _, _ = load_checkpoint(model2, optimizer2, outpath2)
        return model1, model2, criterion1, criterion2, optimizer1, optimizer2, test_loader

def decoras_train_model(hyperparameters, device, name):
    with wandb.init(project='decoras', name=name, entity='bradipo', config=hyperparameters):
        config = wandb.config
        bf_msle, bf_bce, msle, bce, optimizer1, optimizer2, train_loader, valid_loader = make(config, device)
        if name == 'msle_decoras':
            train(bf_msle, train_loader, valid_loader, msle, optimizer1, config, name, device)
        else:
            train(bf_bce, train_loader, valid_loader, bce, optimizer1, config, name, device)

def decoras_test(hyperparameters, device, name):
    with wandb.init(project='hyperparameters', name=name, entity='bradipo', config=hyperparameters):
        config = wandb.config
        bf_msle, bf_bce, msle, bce, optimizer1, optimizer2, test_loader = make(config, device)
        if config['test_selector'] == 'msle':
            tp, n, fp, fn, precision, recall, mean_IoU, true_x, true_y, predicted_x, predicted_y, dIoUs, dfluxes, tIoUs, tfluxes = test_single_model(bf_msle, test_loader, msle, config, device)
        elif config['test_selector'] == 'bce':
            tp, n, fp, fn, precision, recall, mean_IoU, true_x, true_y, predicted_x, predicted_y, dIoUs, dfluxes, tIoUs, tfluxes = test_single_model(bf_bce, test_loader, bce, config, device)
        else:
            tp, n, fp, fn, precision, recall, mean_IoU, true_x, true_y, predicted_x, predicted_y = test(bf_msle, bf_bce, msle, bce, config, device)
    return tp, n, fp, fn, precision, recall, mean_IoU, true_x, true_y, predicted_x, predicted_y, dIoUs, dfluxes, tIoUs, tfluxes

# ------------------------- SCRIPT --------------------------------------
config = dict(
    epochs = 200,
    batch_size = 64,
    multi_gpu = False,
    mode = 'train',
    learning_rate = 0.001,
    weight_decay = 1e-5,
    early_stopping = 'True',
    patience = 20,
    hidden_channels = 1024,
    criterion=['l_1', 'ssim'],
    warm_start=False,
    warm_start_iterations = 3,
    twoD_iou_threshold = 0.8,
    twoD_dist_threshold = 3, 
    data_folder='/lustre/home/mdelliveneri/ALMADL/data/',
    project = 'Decoras',
    output_dir = '/lustre/home/mdelliveneri/ALMADL/trained_models',
    prediction_dir = '/lustre/home/mdelliveneri/ALMADL/predictions',
    plot_dir = '/lustre/home/mdelliveneri/ALMADL/plots',
    model1_name = 'msle_decoras',
    model2_name = 'bce_decoras',
    phase_selector = 'blobsfinder',
    test_selector = 'blobsfinder',
    
)
# Device configuration, later modify to work with multiple GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if config['phase_selector'] == 'msle':
    decoras_train_model(config, device, config['model1_name'])
elif config['phase_selector'] == 'bce':
    decoras_train_model(config, device, config['model2_name'])
elif config['phase_selector'] == 'blobsfinder':
    blobsfinder_train_model(config, device, 'blobsfinder2')
else:
    config['mode'] = 'test'
    outname = os.path.join(config['plot_dir'], 'decoras_prediction.png')
    tp, n, fp, fn, precision, recall, mean_IoU, true_x,  true_y, predicted_x, predicted_y = decoras_test(config)
    print('TP: ', tp)
    print('FP: ', fp)
    print('FN: ', fn)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Mean IoU: ', mean_IoU)
    res_x = true_x - predicted_x
    res_y = true_y - predicted_y
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8 *2, 8 * 2))
    ax[0, 0].hist(res_x, bins=50, edgecolor='black', color='dodgerblue')
    ax[0, 0].set_ylabel('N')
    ax[0, 0].set_xlabel('Residuals x')
    ax[1, 0].scatter(true_x, predicted_x, color='dodgerblue', s=2)
    y_lim = ax[1, 0].get_ylim()
    x_lim = ax[1, 0].get_xlim()
    ax[1, 0].plot(x_lim, y_lim, color = 'r', linestyle='dashed')
    ax[1, 0].set_ylabel('Predicted x')
    ax[1, 0].set_xlabel('True x')

    ax[0, 1].hist(res_y, bins=50, edgecolor='black', color='dodgerblue')
    ax[0, 1].set_ylabel('N')
    ax[0, 1].set_xlabel('Residuals y')
    ax[1, 1].scatter(true_y, predicted_y, color='dodgerblue', s=2)
    y_lim = ax[1, 1].get_ylim()
    x_lim = ax[1, 1].get_xlim()
    ax[1, 1].plot(x_lim, y_lim, color = 'r', linestyle='dashed')
    ax[1, 1].set_ylabel('Predicted y')
    ax[1, 1].set_xlabel('True y')
    plt.title('Decoras Source Detection Plots')
    plt.tight_layout()
    plt.savefig(outname)
    print('Mean Residual x: ', np.mean(res_x))
    print('Std Residual x: ', np.std(res_x))
    print('Mean Residual y: ', np.mean(res_y))
    print('Std Residual y: ', np.std(res_y))

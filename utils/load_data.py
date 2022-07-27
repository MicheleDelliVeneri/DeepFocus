from tkinter import W
from bleach import clean
import numpy as np
import pandas as pd
import os
import sys
from scipy.fftpack import shift
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
import random
from astropy.io import fits
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from torch.utils.data._utils.collate import default_collate
from typing import Sequence, Dict
from tqdm import tqdm
from scipy.ndimage import binary_dilation, generate_binary_structure

HISTORY = "history"

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_sample(sample):
    clean_img = sample['clean_img']
    dirty_img = sample['dirty_img']
    if isinstance(clean_img, torch.Tensor):
        clean_img = clean_img[0].numpy()
        dirty_img = dirty_img[0].numpy()

    boxes = sample['boxes']
    focused = sample['focused']
    dirty_spectra = sample['dirty_spectra']
    if boxes.shape[1] == 5:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        box_cmap = get_cmap(len(boxes), name="tab20")
        im0 = ax[0].imshow(clean_img, origin='lower', cmap='viridis')
        for j, box in enumerate(boxes):
            rect = patches.Rectangle(xy=(box[1], box[0]), 
                    width=box[3] - box[1],
                    height=box[2] - box[0],
                    angle = box[4], 
                    facecolor="none",
                    edgecolor=box_cmap(j),
                    label='Source n. ' + str(j))
            ax[0].add_patch(rect)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('Sky Model Image')
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(dirty_img, origin='lower', cmap='viridis')
        for j, box in enumerate(boxes):
            rect = patches.Rectangle(xy=(box[1], box[0]), 
                    width=box[3] - box[1],
                    height=box[2] - box[0],
                    angle = box[4],
                    facecolor="none",
                    edgecolor=box_cmap(j),
                    label='Source n. ' + str(j))
            ax[1].add_patch(rect)
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_title('Calibrated Dirty Image')
        plt.colorbar(im1, ax=ax[1])
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        box_cmap = get_cmap(len(boxes), name="tab20")
        im0 = ax[0].imshow(clean_img, origin='lower', cmap='viridis')
        for j, box in enumerate(boxes):
            rect = patches.Rectangle(xy=(box[1], box[0]), 
                    width=box[3] - box[1],
                    height=box[2] - box[0], 
                    facecolor="none",
                    edgecolor=box_cmap(j),
                    label='Source n. ' + str(j))
            ax[0].add_patch(rect)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('Sky Model Image')
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(dirty_img, origin='lower', cmap='viridis')
        for j, box in enumerate(boxes):
            rect = patches.Rectangle(xy=(box[1], box[0]), 
                    width=box[3] - box[1],
                    height=box[2] - box[0], 
                    facecolor="none",
                    edgecolor=box_cmap(j),
                    label='Source n. ' + str(j))
            ax[1].add_patch(rect)
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_title('Calibrated Dirty Image')
        plt.colorbar(im1, ax=ax[1])
        plt.tight_layout()
        plt.show()
    """

    if boxes.shape[0] <= 3:
        fig, ax = fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3 * 8, 8))
        for i, _ in enumerate(boxes):
            ax[i].imshow(focused[i], cmap='magma', origin='lower')
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_title('Focused Source {}'.format(str(i)))
        dif = 3 - len(boxes)
        if dif > 0:
            fig.delaxes(ax[2])
        plt.tight_layout()
        plt.show()

        fig, ax = fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3 * 8, 4))
        for i, _ in enumerate(boxes):
            ax[i].plot(dirty_spectra[i])
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_title('Dirty Spectrum {}'.format(str(i)))
        dif = 3 - len(boxes)
        if dif > 0:
            fig.delaxes(ax[2])
        plt.tight_layout()
        plt.show()
    


    else:
        fig, ax = fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(3 * 8, 2 * 8))
        for i, _ in enumerate(boxes):
            if i <= 2:
                ax[0, i].imshow(focused[i], cmap='magma', origin='lower')
                ax[0, i].set_xlabel('x')
                ax[0, i].set_ylabel('y')
                ax[0, i].set_title('Focused Source {}'.format(str(i)))
            else:

                ax[1, i - 3].imshow(focused[i], cmap='magma', origin='lower')
                ax[1, i - 3].set_xlabel('x')
                ax[1, i - 3].set_ylabel('y')
                ax[1, i - 3].set_title('Focused Source {}'.format(str(i)))
        
        dif = 6 - len(boxes)
        t = 2
        while dif > 0:
            fig.delaxes(ax[1][t])
            dif -= 1
            t -= 1
        plt.tight_layout()
        plt.show()

        fig, ax = fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(3 * 8, 2 * 4))
        for i, _ in enumerate(boxes):
            if i <= 2:
                ax[0, i].plot(dirty_spectra[i])
                ax[0, i].set_xlabel('x')
                ax[0, i].set_ylabel('y')
                ax[0, i].set_title('Dirty Spectrum {}'.format(str(i)))
            else:

                ax[1, i - 3].plot(dirty_spectra[i])
                ax[1, i - 3].set_xlabel('x')
                ax[1, i - 3].set_ylabel('y')
                ax[1, i - 3].set_title('Dirty Spectrum {}'.format(str(i)))
        dif = 6 - len(boxes)
        t = 2
        while dif > 0:
            fig.delaxes(ax[1][t])
            dif -= 1
            t -= 1
        plt.tight_layout()
        plt.show()
    """

def get_corners(bboxes):
    
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.
    
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated

def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
        
def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    
    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]


    return bbox

def rotate_img(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image

class NormalizeImage(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        dirty_image = sample['dirty_img']
        clean_image = sample['clean_img']
        cmin, cmax = np.min(clean_image), np.max(clean_image)
        dmin, dmax = np.min(dirty_image), np.max(dirty_image)
        dirty_image = (dirty_image - dmin) / (dmax - dmin)
        clean_image = (clean_image - cmin) / (cmax - cmin)
        sample['dirty_img'] = dirty_image
        sample['clean_img'] = clean_image
        return sample

class RandomRotate(object):
    """
    Rotate the image and bounding boxes in a sample
    """
    def __init__(self):
        pass
    
    def __call__(self, sample):
        dirty_image = sample['dirty_img']
        clean_image = sample['clean_img']
        boxes = sample['boxes'][:, [1, 0, 3, 2]]
        angle = random.randint(-180, 180)
        w,h = dirty_image.shape[1], dirty_image.shape[0]
        cx, cy = w//2, h//2
        clean_image = rotate_img(clean_image, angle)
        dirty_image = rotate_img(dirty_image, angle)
        corners = get_corners(boxes)
        corners = np.hstack((corners, boxes[:,4:]))
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        boxes = get_enclosing_box(corners)
        angles = np.repeat(angle, len(boxes)).reshape(-1, 1)
        boxes = np.hstack((boxes, angles))
        sample['boxes'] = boxes[:, [1, 0, 3, 2, 4]]
        sample['dirty_img'] = dirty_image
        sample['clean_img'] = clean_image
        return sample
        
class RandomHorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
            dirty_image = sample['dirty_img']
            clean_image = sample['clean_img']
            boxes = sample['boxes'][:, [1, 0, 3, 2]]
            img_center = np.array(dirty_image.shape)/2
            img_center = np.hstack((img_center, img_center))
            if random.random() < self.p:
                dirty_image = dirty_image[:, ::-1]
                clean_image = clean_image[:, ::-1]
                #boxes[:, [0, 2]] += 2*(img_center[[0, 2]] - boxes[:, [0, 2]])
                boxes[:, [0, 2]] = np.add(
                    boxes[:, [0, 2]], 2*(img_center[[0, 2]] - boxes[:, [0, 2]]),
                    out=boxes[:, [0, 2]], casting='unsafe')
                box_w = abs(boxes[:, 0] - boxes[:, 2])

                boxes[:, 0] -= box_w
                boxes[:, 2] += box_w
                sample['boxes'] = boxes[:, [1, 0, 3, 2]]
                sample['dirty_img'] = dirty_image
                sample['clean_img'] = clean_image
            return sample

class RandomVerticalFlip(object):

    """Randomly vertically flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
            dirty_image = sample['dirty_img']
            clean_image = sample['clean_img']
            boxes = sample['boxes'][:, [1, 0, 3, 2]]
            img_center = np.array(dirty_image.shape)/2
            img_center = np.hstack((img_center, img_center))
            if random.random() < self.p:
                dirty_image = dirty_image[::-1, :]
                clean_image = clean_image[::-1, :]
                #boxes[:, [0, 2]] += 2*(img_center[[0, 2]] - boxes[:, [0, 2]])
                boxes[:, [1, 3]] = np.add(
                    boxes[:, [1, 3]], 2*(img_center[[1, 3]] - boxes[:, [1, 3]]),
                    out=boxes[:, [1, 3]], casting='unsafe')
                box_h = abs(boxes[:, 1] - boxes[:, 3])

                boxes[:, 1] -= box_h
                boxes[:, 3] += box_h
                sample['boxes'] = boxes[:, [1, 0, 3, 2]]
                sample['dirty_img'] = dirty_image
                sample['clean_img'] = clean_image
            return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensor"""

    def __init__(self):
        pass

    def __call__(self, sample):
            dirty_image = torch.from_numpy(sample['dirty_img'][np.newaxis, :, :])
            clean_image = torch.from_numpy(sample['clean_img'][np.newaxis, : , :])
            sample['dirty_img'] = dirty_image
            sample['clean_img'] = clean_image
            return sample

class Crop(object):
    """
    Crop the image and bounding boxes in a sample
    
    Args: output_size (int)
    """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
    def __call__(self, sample):
        dirty_image = sample['dirty_img']
        clean_image = sample['clean_img']
        h, w = dirty_image.shape
        h = h // 2
        w = w // 2
        new_h, new_w = self.output_size
        new_h = new_h // 2
        new_w = new_w // 2
        dirty_image = dirty_image[h - new_h: h + new_h, w - new_w: w + new_w]
        clean_image = clean_image[h - new_h: h + new_h, w - new_w: w + new_w]
        d_x = h - new_h
        d_y = w - new_w
        boxes = sample['boxes']
        y_0, x_0, y_1, x_1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes = np.array([y_0 - d_y, x_0 - d_x, y_1 - d_y, x_1 - d_x]).T
        sample['boxes'] = boxes
        sample['dirty_img'] = dirty_image
        sample['clean_img'] = clean_image
        return sample

class ALMADataset(Dataset):
    """ALMA Dataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args: 
            csv_file (string): Path with the CSV file containing sources parameters
            root_dir (string): Directory containing the .fits files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.parameters = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform
        self.dirty_list = np.array(natsorted([
            os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if 'dirty' in file]))
        self.clean_list = np.array(natsorted([
            os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if 'clean' in file]))
    def __len__(self):
        return len(self.dirty_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()      
        dirty_name = self.dirty_list[idx]
        clean_name = self.clean_list[idx]
        db_idx = float(clean_name.split('_')[-1].split('.')[0])
        params = self.parameters.loc[self.parameters.ID == db_idx]
        boxes = np.array(params[["y0", "x0", "y1", "x1"]].values)
        snrs = params['snr'].values
        clean_img = np.sum(fits.getdata(clean_name)[0], axis=0)
        dirty_img = np.sum(fits.getdata(dirty_name)[0], axis=0)
        
        sample = {'dirty_img': dirty_img, 'clean_img': clean_img, 'boxes': boxes, 'snrs': snrs, 'idx': idx}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        """
        clean_images = list()
        dirty_images = list()
        boxes = list()
        snrs = list()
        idxs = list()
        for b in batch:
            clean_images.append(b['clean_img'])
            dirty_images.append(b['dirty_img'])
            boxes.append(b['boxes'])
            snrs.append(b['snrs'])
            idxs.append(b['idx'])
            
        clean_images = torch.stack(clean_images, dim=0)
        dirty_images = torch.stack(dirty_images, dim=0)
           
        return dirty_images, clean_images, boxes, snrs

class PipelineDataLoader(object):
    def __init__(self, csv_file, root_dir):
        """
        Args: 
            csv_file (string): Path with the CSV file containing sources parameters
            root_dir (string): Directory containing the .fits files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.parameters = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.dirty_list = np.array(natsorted([
            os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if 'dirty' in file]))
        self.clean_list = np.array(natsorted([
            os.path.join(self.root_dir, file) for file in os.listdir(self.root_dir) if 'clean' in file]))
    def create_dataset(self):
        focused = []
        line_images = []
        targs = []
        spectra = []
        dspectra = []
        for idx in tqdm(range(len(self.dirty_list))):
            dirty_name = self.dirty_list[idx]
            clean_name = self.clean_list[idx]
            db_idx = float(dirty_name.split('_')[-1].split('.')[0])
            params = self.parameters.loc[self.parameters.ID == db_idx]
            boxes = np.array(params[["y0", "x0", "y1", "x1"]].values)
            z_ = np.array(params["z"].values)
            fwhm_z = np.array(params["fwhm_z"].values)
            zboxes = np.array([z_ - fwhm_z, z_ + fwhm_z]).astype(int).T
            targets = np.array(params[["ID", "x", "y", "z", "fwhm_x", "fwhm_y", "pa", "flux", 'continuum']].values)
            targets['extensions'] = 2 * params['fwhm_z']
            dirty_cube = fits.getdata(dirty_name) 
            clean_cube = fits.getdata(clean_name)
            snrs = params['snr'].values
            dirty_spectra = np.array([
                np.sum(dirty_cube[0][:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])
            clean_spectra = np.array([
                np.sum(clean_cube[0][:, boxes[j][0]: boxes[j][2], boxes[j][1]: boxes[j][3]], axis=(1, 2))
                for j in range(len(boxes))
                ])

            xs = []
            ys = []
            for i, box in enumerate(boxes):
                y_0, x_0, y_1, x_1 = box
                z_0, z_1 = zboxes[i]
                width_x, width_y = x_1 - x_0, y_1 - y_0
                x, y = x_0 + 0.5 * width_x, y_0 + 0.5 * width_y
                source = np.sum(dirty_cube[0][z_0:z_1, int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                cont_image = np.mean(np.concatenate(
                                    (
                                        dirty_cube[0][:z_0, int(y) - 32: int(y) + 32,
                                                       int(x) - 32: int(x) + 32],
                                        dirty_cube[0][z_1:, int(y) - 32: int(y) + 32,
                                                       int(x) - 32: int(x) + 32]
                                    ),
                                    axis=0), axis=0) * (z_1 - z_0)

                
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
                cont_source = cont_image[left:right, bottom:top]
                cont_subtracted = source  - cont_source
                model = np.sum(clean_cube[0][z_0:z_1, int(y) - 32: int(y) + 32, int(x) - 32: int(x) + 32], axis=0)
                min_, max_ = np.min(model), np.max(model)
                model = (model - min_) / (max_ - min_)
                model[model > 0.15] = 1
                model = model.astype(int)
                struct = generate_binary_structure(2, 2)
                model = binary_dilation(model, struct)
                line_image = model * cont_subtracted
                min_, max_ = np.min(line_image), np.max(line_image)
                line_image = (line_image - min_) / (max_ - min_)
                min_, max_ = np.min(source), np.max(source)
                source = (source - min_) / (max_ - min_)
                focused.append(source[np.newaxis])
                line_images.append(line_image[np.newaxis])
                xs.append(32 - left)
                ys.append(32 - bottom)
            targets[:, 1] = xs
            targets[:, 2] = ys
            for j in range(len(targets)):
                targs.append(targets[j])
                dspec = dirty_spectra[j]
                dspec = (dspec - np.mean(dspec)) / np.std(dspec)
                spec = clean_spectra[j]
                spec = (spec - np.mean(spec)) / np.std(spec)
                spectra.append(spec)
                dspectra.append(dspec)
        focused = np.array(focused)
        line_images = np.array(line_images)
        targs = np.array(targs)   
        spectra = np.array(spectra)
        dspectra = np.array(dspectra)
        dspectra = np.transpose(dspectra[np.newaxis], 
                                        axes=(1, 2, 0))
        spectra = np.transpose(spectra[np.newaxis], 
                                        axes=(1, 2, 0)) 

        return spectra, dspectra, focused, targs, line_images    

    
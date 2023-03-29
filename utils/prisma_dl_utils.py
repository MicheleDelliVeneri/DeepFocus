from astropy.io.fits import getheader
from astropy.io import fits
import numpy as np
import torch
from tqdm import tqdm
from itertools import starmap
from torch.utils.data import Dataset 
from scipy.ndimage import rotate
import rasterio
import patchify
from torchvision import transforms


###### the PRISMA preprocessing datacube 

def prisma_preprocess(path: str):
  #convert tif file in numpy array
  numpy_array = rasterio.open(path).read()
  # rotate the image
  lst=[]
  for i in range (numpy_array.shape[0]):
    lst.append(rotate(numpy_array[i,:,:], angle=14.1, reshape=False, order=0))
  numpy_array = np.array(lst)
  #cut the black stripes
  numpy_array = numpy_array [:,150:1100, 150:1100]
  # if the array is not a mask, cut the absorption bands due to the atmosphere
  if (numpy_array.shape[0] > 1):
    numpy_array = np.concatenate((numpy_array[0:95], numpy_array[109:143], numpy_array[161:]), axis=0)
  return numpy_array
 
def create_dataset(numpy_datacube, patch_bands, patch_height, patch_width, patch_overlap):
  data_patch = patchify.patchify(numpy_datacube, (patch_bands,patch_height,patch_width), patch_overlap)
  dataset = data_patch.reshape(data_patch.shape[0] * data_patch.shape[1]* data_patch.shape[2] , data_patch.shape[3], data_patch.shape[4], data_patch.shape[5])
  return dataset   

class PrismaDataset(Dataset):
  def __init__(self, datacubes, masks, transforms = None):
    # pass the datacube and mask previously built
    self.datacubes = datacubes
    self.masks = masks
    self.transforms = transforms
  
  def __len__(self):
    # return the number of total samples contained in the dataset
    return len(self.datacubes)

  def __getitem__(self, idx):
    datacube = self.datacubes[idx]
    mask = self.masks[idx]

    if self.transforms is not None:
      datacube = self.transforms(datacube.reshape(datacube.shape[1], datacube.shape[2], datacube.shape[0]))
      mask =  self.transforms(mask.reshape(mask.shape[1], mask.shape[2], mask.shape[0]))
    else:
      datacube = torch.from_numpy(datacube)
      mask = torch.from_numpy(mask)

    return (datacube, mask)

transforms  = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=1)
                ])	
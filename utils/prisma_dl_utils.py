# %%
import numpy as np
import torch
from tqdm import tqdm
from itertools import starmap
from torch.utils.data import Dataset 
from scipy.ndimage import rotate
import rasterio
import patchify
from torchvision import transforms
import matplotlib.pyplot as plt
import torchio as tio
import os
from torch.utils.data import DataLoader

# %%
def prisma_preprocess(images_path: str, training_path: str, file_name : str, train_num : int):
  """
  Loads the PRISMA datacubes and preprocess them to be used in the training. Loads from images_path as a numpy 
  array, rotates each slice, cuts the black stripes, cuts the absorption bands due to the atmosphere and saves.
  """
  #convert tif file in numpy array
  numpy_array = rasterio.open(images_path).read() # type: ignore
  # rotate the image
  lst=[]
  for i in range (numpy_array.shape[0]):
    lst.append(rotate(numpy_array[i,:,:], angle=14.1, reshape=False, order=0))
  numpy_array = np.array(lst)
  del lst
  #cut the black stripes
  numpy_array = numpy_array [:,150:1100, 150:1100]
  # if the array is not a mask, cut the absorption bands due to the atmosphere
  if (numpy_array.shape[0] > 1):
    numpy_array = np.concatenate((numpy_array[0:95], numpy_array[109:143], numpy_array[161:]), axis=0)
    if (train_num > 1):
      for i in range (0, 360 + 360//train_num, 360//train_num):
        numpy_array_rot = np.array([rotate(numpy_array[j,:,:], i, mode='reflect', reshape=False, order=0) for j in range (numpy_array.shape[0])])
        numpy_array_rot = np.transpose(numpy_array_rot, (2,1,0))
        numpy_array_rot = np.expand_dims(numpy_array_rot, axis=0)
        torch.save(torch.from_numpy(numpy_array_rot), training_path + file_name + str(i) + '.pt')
        del numpy_array_rot
    else:    
      numpy_array = np.transpose(numpy_array, (2,1,0))
      numpy_array = np.expand_dims(numpy_array, axis=0)
      torch.save(torch.from_numpy(numpy_array), training_path + file_name + '.pt')
    del numpy_array
  return(print("the PRISMA datacubes are preporcessed and saved"))


# %%
def torch_reader(path):
    data = torch.load(path)
    affine = None
    return data, affine
# %%
subject_lst = []
for file in os.listdir(training_path):
    file_path = os.path.join(training_path, file)
    subject = tio.Subject(
        image = tio.ScalarImage(file_path, reader=torch_reader)
    )
    subject_lst.append(subject)



# %%
sardinia_dataset = tio.SubjectsDataset(subject_lst)
# %%
patch_size = 96
queue_length = 300
samples_per_volume = 10
sampler = tio.data.UniformSampler(patch_size)
patches_queue = tio.Queue(
    sardinia_dataset,
    queue_length,
    samples_per_volume,
    sampler,
    num_workers=4,
)
# %%
patches_loader = DataLoader(
    patches_queue,
    batch_size=16,
    num_workers=0,  # this must be 0
)
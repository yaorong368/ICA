import torch
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import argparse

import ica

from numpy.random import permutation
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR





data_id = os.listdir('/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/')
path_list = []
for i in data_id:
    nii_path = '/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/' + i + '/ses_01/func/SM.nii'
    path_list.append(nii_path)

def norm(x):
    """Computes the norm of a vector or the Frobenius norm of a
    matrix_rank
    """
    return torch.norm(torch.flatten(x))

def get_mixture(path_list, dims_remain):
    print('getting file:', 0)
    mri = nib.load(path_list[0])
    data = np.asanyarray(mri.dataobj)
    list_x = []
    for j in range(int(data.shape[-1])):
        list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))
    mixture = np.concatenate(list_x,axis=1).T
    
    for i in range(1, 50):
        print('getting file:', i)
        mri = nib.load(path_list[i])
        data = np.asanyarray(mri.dataobj)
        list_x = []
        for j in range(int(data.shape[-1])):
            list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))

        conc_x = np.concatenate(list_x,axis=1)
        mixture = np.concatenate((mixture, conc_x.T), axis=0)
        
    del mri, data, list_x, conc_x
#     mixture = np.float64(mixture.T)
    print('the shape of mixture is:', mixture.shape)
    print('whitening')
    mixture, white, dewhite = ica.pca_whiten(mixture, dims_remain)
    mixture = torch.from_numpy(mixture).type(torch.float32)
    print('done')
    return mixture, white, dewhite

mixture, white, dewhite = get_mixture(path_list, 50)
np.save('/data/users2/yxiao11/model/ICA/mri_data/mixture.npy', mixture)
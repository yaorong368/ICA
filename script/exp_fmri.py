#this is the most updated code for infonce training with 50 subjects and compressed to 50 components
#can get competitable result compared wiht infomax

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


import sys
sys.path.append('/data/users2/yxiao11/model/ICA')
from modules.util import entropyLoss, getdata, infomaxICA, entropy



dims_remain=50

data_id = os.listdir('/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/')
path_list = []
for i in data_id:
    nii_path = '/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/' + i + '/ses_01/func/SM.nii'
    path_list.append(nii_path)

b = nib.load(path_list[2])

mixture = torch.from_numpy(np.load('/data/users2/yxiao11/model/ICA/mri_data/masked_mixture.npy')).type(torch.float32)
msk_idx = np.nonzero(np.asanyarray(nib.load('/data/users2/yxiao11/model/ICA/mri_data/mask.nii').dataobj).flatten())[0]
IPT = mixture.type(torch.float32)

icaica1 = (np.asanyarray(nib.load('/data/users2/yxiao11/model/ICA/mri_data/icaica1_50.nii').dataobj).reshape(int(53*63*52), 50))[msk_idx, :]

print('the input data shape:', mixture.shape)
# batch_size = int(np.floor(np.sqrt(IPT.shape[1] / 3)))
batch_size = 50
print(batch_size)
num_epoch = 50
# learning_rate = 0.1 / np.log(dims_remain)
learning_rate = 0.1

device = torch.device('cuda:0')
model = infomaxICA(dims_remain)
dataset = getdata(IPT)
# sampler = SequentialSampler(dataset)
sampler = RandomSampler(dataset)
loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=learning_rate,
                        # eps=10e-10,
                        # weight_decay = 1e-9,
                        )

scheduler = CosineAnnealingLR(optimizer, T_max = num_epoch*2)

model = model.to(device)
IPT = IPT.to(device)

#------customize loss-------------

 
def cor(opt):
    _, d = opt.shape
    index = torch.triu_indices(d, d, offset=1)
    cc = torch.square(torch.matmul(opt.T,opt))
    return torch.sum(cc[index[0], index[1]])
#-------------------------------

loss_tracker = []

def norm(x):
    """Computes the norm of a vector or the Frobenius norm of a
    matrix_rank
    """
    return torch.norm(torch.flatten(x))


step0 = 0
angle_delta = 0
print('start training')
for i in range(num_epoch):
    LOSS = 0
    for step, ipt in enumerate(loader):
        model.zero_grad()
        ipt = ipt.to(device)
        
        opt = model.forward(ipt)

        loss = cor(opt)
        loss.backward()
        optimizer.step()
        LOSS+=loss
        
    LOSS = LOSS.cpu()
    loss_tracker.append(LOSS.detach().numpy()/len(loader.sampler))
    if step0 % 3 == 0:
        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1)
        plt.plot(loss_tracker)
    
        # data_opt = model.forward(IPT.T).cpu().detach().numpy()
        data_opt = (model.W1.weight.data@IPT).cpu().detach().numpy()
        # data2 = angle2cart(data).cpu().detach().numpy()
#         data2 = data.cpu().detach().numpy()
        
        plt.subplot(1,3,2)
        plt.plot(data_opt[0], data_opt[1], '.', ms=0.5)

        plt.subplot(1,3,3)
        plt.imshow(abs(np.corrcoef(data_opt, icaica1.T)), cmap='gist_heat')

        plt.savefig('/data/users2/yxiao11/model/ICA/figures/experiments_50fmri.png')
        plt.close()


    if step0%10 == 0:
        data2 = model.W1.weight.data @ IPT
        data2 = data2.cpu().detach().numpy()
        scale = data2.std(axis=1).reshape((-1, 1))
        data2 = data2 / scale
       
        my_nifiti = np.zeros([dims_remain, 53*63*52])
        my_nifiti[:, msk_idx] = data2
        my_nifiti = my_nifiti.T
        nifiti = my_nifiti.reshape(53,63,52,dims_remain)
        new_image = nib.Nifti1Image(nifiti, affine=b.affine, header=b.header)
        nib.save(new_image, '/data/users2/yxiao11/model/ICA/mri_data/cc_50.nii')

    step0 += 1
    print(step0, LOSS.detach().numpy()/len(loader.sampler))
    scheduler.step()


model.cpu()
IPT = IPT.cpu()

data2 = model.W1.weight.data @ IPT
data2 = data2.detach().numpy()
scale = data2.std(axis=1).reshape((-1, 1))
data2 = data2 / scale

my_nifiti = np.zeros([dims_remain, 53*63*52])
my_nifiti[:, msk_idx] = data2
my_nifiti = my_nifiti.T
nifiti = my_nifiti.reshape(53,63,52,dims_remain)

# nifiti = np.array(a)
new_image = nib.Nifti1Image(nifiti, affine=b.affine, header=b.header)
nib.save(new_image, '/data/users2/yxiao11/model/ICA/mri_data/cc_50.nii')


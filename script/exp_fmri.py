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


# def get_mixture(path_list, dims_remain):
#     print('getting file:', 0)
#     mri = nib.load(path_list[0])
#     data = np.asanyarray(mri.dataobj)
#     list_x = []
#     for j in range(int(data.shape[-1])):
#         list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))
#     mixture = np.concatenate(list_x,axis=1).T
    
#     for i in range(1, 100):
#         print('getting file:', i)
#         mri = nib.load(path_list[i])
#         data = np.asanyarray(mri.dataobj)
#         list_x = []
#         for j in range(int(data.shape[-1])):
#             list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))

#         conc_x = np.concatenate(list_x,axis=1)
#         mixture = np.concatenate((mixture, conc_x.T), axis=0)
        
#     del mri, data, list_x, conc_x
# #     mixture = np.float64(mixture.T)
#     print('the shape of mixture is:', mixture.shape)
#     print('whitening')
#     mixture, white, dewhite = ica.pca_whiten(mixture, dims_remain)
#     mixture = torch.from_numpy(mixture).type(torch.float32)
#     print('done')
#     return mixture, white, dewhite

dims_remain=50

data_id = os.listdir('/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/')
path_list = []
for i in data_id:
    nii_path = '/data/qneuromark/Data/FBIRN/Data_BIDS/Raw_Data/' + i + '/ses_01/func/SM.nii'
    path_list.append(nii_path)

b = nib.load(path_list[2])

mixture = torch.from_numpy(np.load('/data/users2/yxiao11/model/ICA/mri_data/mixture.npy')).type(torch.float32)
# mixture, white, dewhite = get_mixture(path_list, dims_remain)
# IPT = torch.from_numpy(mixture.copy()).type(torch.float32)
IPT = mixture.type(torch.float32)

print('the input data shape:', mixture.shape)
batch_size = int(np.floor(np.sqrt(IPT.shape[1] / 3)))
# batch_size = 512
print(batch_size)
num_epoch = 200
learning_rate = 0.001 / np.log(dims_remain)
# learning_rate = 0.9

device = torch.device('cuda:0')
model = infomaxICA(dims_remain)
dataset = getdata(IPT)
sampler = SequentialSampler(dataset)
# sampler = RandomSampler(dataset)
loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
#                            lr=learning_rate,
#                            )
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=learning_rate,
                        eps=10e-4,
                        weight_decay = 1e-4,
                        )

# scheduler = CosineAnnealingLR(optimizer, T_max = num_epoch)
scheduler = OneCycleLR(optimizer, 
                       learning_rate, 
                       steps_per_epoch=len(loader), 
                       epochs=num_epoch,
                      )

model = model.to(device)
IPT = IPT.to(device)

#------customize loss-------------
def my_loss(data):
    """
    Compute the kurtosis of a set of data using PyTorch.

    Args:
    data (torch.Tensor): A 1D tensor containing the data.

    Returns:
    float: The kurtosis of the data.
    """
    # Calculate the mean and standard deviation of the data
    mean = torch.mean(data, axis=0)
    std = torch.std(data, axis=0)

    # Calculate the fourth central moment (raw kurtosis)
    fourth_moment = torch.mean((data - mean)**4, axis=0)

    # Calculate the kurtosis using the fourth central moment
    n = data.size(0)
    kurtosis = fourth_moment / (std**4) - 3.0

    return torch.mean(-kurtosis**2)
# def infonce(opt, ipt_batch, t=100):
    
#     _, d = opt.shape
#     Ln = 0
#     for i in range(d):
        
#         index = [x for x in range(d) if x!=i]
#         a = [torch.exp(torch.norm(ipt_batch.T@opt[:, k:k+1])/t) for k in index]
#         denominator = torch.stack(a).sum() + 1e-12

#         numerator = torch.exp(torch.norm(ipt_batch.T@opt[:, i:i+1])/t)
#         Ln += torch.log(numerator/denominator)
#     return Ln.sum()/d

def infonce(opt, ipt_batch, t=100):
    
    _, d = opt.shape
    
    f = torch.exp(torch.norm(ipt_batch.T@opt, dim=0)/t)
    Ln = 0
    for i in range(d):
        index = [x for x in range(d) if x!=i]
        denominator = f[index].sum() + 1e-12

        numerator = f[i]
        Ln += torch.log(numerator/denominator)
    return Ln.sum()/d

# def infonce(opt, ipt_batch, t=100):
    
#     _, d = opt.shape
    
#     numerators = torch.exp(torch.norm(ipt_batch.T@opt, dim=0)/t)
#     denominator = numerators.sum() + 1e-12
    
#     return torch.log(numerators/denominator).sum()
#-------------------------------

loss_tracker = []

def norm(x):
    """Computes the norm of a vector or the Frobenius norm of a
    matrix_rank
    """
    return torch.norm(torch.flatten(x))


k = int(batch_size*0.16)

change = 100
W_STOP = 1e-7
d_weigths = torch.zeros(dims_remain)
old_d_weights = torch.zeros([1, dims_remain])
torch_pi = torch.acos(torch.zeros(1)).item() * 2 

# for epoch in range(num_epoch):
step0 = 0
angle_delta = 0
print('start training')
# while step0 < num_epoch and change > W_STOP:
for i in range(num_epoch):
    LOSS = 0
    old_weight = torch.clone(model.W1.weight.data)
    for step, ipt in enumerate(loader):
        model.zero_grad()
        ipt = ipt.to(device)
        
        opt = model.forward(ipt)

        # loss = entropy(opt, k=k, dis=1)
#         loss = my_loss(opt)
        loss = infonce(opt, ipt)
        loss.backward()
        optimizer.step()
        LOSS+=loss
    d_weigths = model.W1.weight.data - old_weight
    change = norm(d_weigths)**2
    d_weigths = d_weigths.cpu()
    if step > 2:
        angle_delta = torch.arccos(
                    torch.sum(d_weigths * old_d_weights) /
                    (norm(d_weigths) * norm(old_d_weights) + 1e-8)
                ) * 180 / torch_pi
    if angle_delta > 60:
        learning_rate = learning_rate * 0.9
        
        print('lr changed to:', learning_rate)
        
        optimizer.param_groups[0]['lr'] = learning_rate
        old_d_weights = torch.clone(d_weigths)
    elif step == 1:
        old_d_weights = torch.clone(d_weigths) 
        
    LOSS = LOSS.cpu()
    loss_tracker.append(LOSS.detach().numpy()/len(loader.sampler))
    if step0 % 3 == 0:
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(loss_tracker)
    
        data_opt = model.forward(IPT.T).cpu().detach().numpy()
        # data_w = (model.W1.weight.data@IPT).cpu().detach().numpy().T
        # data2 = angle2cart(data).cpu().detach().numpy()
#         data2 = data.cpu().detach().numpy()
        
        plt.subplot(1,2,2)
        plt.plot(data_opt[:,0], data_opt[:,1], '.', ms=0.5)

        # plt.subplot(1,3,3)
        # plt.plot(data_w[:,0], data_w[:, 1] ,'.', ms=0.5)

        plt.savefig('/data/users2/yxiao11/model/ICA/figures/experiments_50fmri.png')
        plt.close()


    if step0%10 == 0:
        data2 = model.W1.weight.data @ IPT
        data2 = data2.cpu()
        scale = data2.std(axis=1).reshape((-1, 1))
        data2 = data2 / scale
        a = data2.permute(1, 0)
        a = a.reshape(53,63,52,dims_remain)
        nifiti = np.array(a)
        new_image = nib.Nifti1Image(nifiti, affine=b.affine, header=b.header)
        # nib.save(new_image, 'mri_data/adam_meanNN_' + str(dims_remain) +'_' + str(batch_size) + 'batch')
        nib.save(new_image, '/data/users2/yxiao11/model/ICA/mri_data/infonce_50.nii')

    step0 += 1
    print(step0, angle_delta, LOSS.detach().numpy()/len(loader.sampler), change)
    scheduler.step()


model.cpu()
IPT = IPT.cpu()

data2 = model.W1.weight.data @ IPT

scale = data2.std(axis=1).reshape((-1, 1))
data2 = data2 / scale
a = data2.permute(1, 0)
a = a.reshape(53,63,52,dims_remain)




nifiti = np.array(a)
new_image = nib.Nifti1Image(nifiti, affine=b.affine, header=b.header)
# nib.save(new_image, 'mri_data/adam_meanNN_' + str(dims_remain) +'_' + str(batch_size) + 'batch')
nib.save(new_image, '/data/users2/yxiao11/model/ICA/mri_data/infonce_50.nii')

#-------ICA-----------

# def get_mixture_ica(path_list, dims_remain):
#     print('getting file:', 0)
#     mri = nib.load(path_list[0])
#     data = np.asanyarray(mri.dataobj)
#     list_x = []
#     for j in range(int(data.shape[-1])):
#         list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))
#     mixture = np.concatenate(list_x,axis=1).T
    

#     for i in range(50):
#         print('getting file:', i)
#         mri = nib.load(path_list[i])
#         data = np.asanyarray(mri.dataobj)
#         list_x = []
#         for j in range(int(data.shape[-1])):
#             list_x.append(np.expand_dims(data[:,:,:,j].flatten(),1))

#         conc_x = np.concatenate(list_x,axis=1)
#         mixture = np.concatenate((mixture, conc_x.T), axis=0)
#     return mixture

# mixture = get_mixture_ica(path_list, dims_remain)
# mixer, mixture, unmixer = ica.ica1(mixture, ncomp=dims_remain, verbose=True)

# nifiti = np.array(mixture).T.reshape(53,63,52,dims_remain)
# new_image = nib.Nifti1Image(nifiti, affine=np.eye(4))
# nib.save(new_image, './mri_data/infonce.nii')
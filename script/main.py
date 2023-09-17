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

from util import entropyLoss, getdata, infomaxICA

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
    
    for i in range(1,200):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameters of ICAtraining')
    parser.add_argument('-s', type=int, default=50,
                        help='number of dimensions remained')
    parser.add_argument('-b', type=int, default=50,
                        help='batch size')
    parser.add_argument('-epoch', type=int, default=500,
                        help='number of epoches')
    args = parser.parse_args()

    dims_remain=args.s
    mixture = torch.from_numpy(np.load('mixture.npy')).type(torch.float32)
    # mixture, white, dewhite = get_mixture(path_list, dims_remain)
    print('the input data shape:', mixture.shape)
    batch_size = int(np.floor(np.sqrt(mixture.shape[1] / 3)/5))
    # batch_size = args.b
    num_epoch = args.epoch
    learning_rate = 0.005 / np.log(dims_remain)
    # learning_rate = 0.9

    device = torch.device('cuda:0')
    model = infomaxICA(dims_remain)

    dataset = getdata(mixture)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate,
                             eps=10e-4,
                             weight_decay = 1e-5,
                            )


    model = model.to(device)

    loss_tracker = []
    change = 100
    W_STOP = 1e-6
    d_weigths = torch.zeros(dims_remain)
    old_weights = torch.eye(dims_remain).to(device)
    old_d_weights = torch.zeros([dims_remain, dims_remain])
    torch_pi = torch.acos(torch.zeros(1)).item() * 2 
    step0 = 0
    # step1 = 0

    print('the batch size is:', batch_size)
    print('the learning rate start with:', learning_rate)

    while step0 < num_epoch and change > W_STOP:
        LOSS = 0
        
        for step, ipt in enumerate(loader):
            model.zero_grad()
            ipt = ipt.to(device)
            
            opt = model.forward(ipt)
    #         loss = entropyLoss(opt.T, device=device)
            loss = entropyLoss(opt.permute(1,0), device=device)
            loss.backward()
            optimizer.step()
        
            LOSS+=loss
        
        d_weigths = model.W2.weight.data - old_weights
        change = norm(d_weigths)**2
        d_weigths = d_weigths.cpu()
        if step > 2:
            angle_delta = torch.arccos(
                        torch.sum(d_weigths * old_d_weights) /
                        (norm(d_weigths) * norm(old_d_weights) + 1e-8)
                    ) * 180 / torch_pi
            # step1 += 1
        old_weights = torch.clone(model.W2.weight.data)

        if angle_delta > 30:
            learning_rate = learning_rate * 0.9
            print('lr decreasedd to: ', learning_rate)
            optimizer.param_groups[0]['lr'] = learning_rate
            old_d_weights = torch.clone(d_weigths)

        # elif step1 > 15:
        #     learning_rate = learning_rate * 0.9
        #     print('lr decreasedd to: ', learning_rate)
        #     optimizer.param_groups[0]['lr'] = learning_rate
        #     old_d_weights = torch.clone(d_weigths)
        #     step1 = 0

        elif step == 1:
            old_d_weights = torch.clone(d_weigths)
        
        
        LOSS = LOSS.cpu()
        loss_tracker.append(LOSS.detach().numpy()/len(loader))
        step0 += 1
        print(step0, angle_delta, LOSS.detach().numpy()/len(loader), change)

    print(learning_rate)
    torch.save(model.state_dict(), 'MeanNN.pt')

    plt.plot(loss_tracker)
    plt.savefig('./loss.png')

    model.cpu()
    mixture = mixture.cpu()

    data2 = model.W2.weight.data @ mixture

    scale = data2.std(axis=1).reshape((-1, 1))
    data2 = data2 / scale
    a = data2.permute(1, 0)
    a = a.reshape(53,63,52,dims_remain)

    nifiti = np.array(a)
    b = nib.load(path_list[2])
    new_image = nib.Nifti1Image(nifiti, affine=b.affine, header=b.header )
    nib.save(new_image, 'mri_data/meanNN_' + str(dims_remain) +'_' + str(batch_size) + 'batch')
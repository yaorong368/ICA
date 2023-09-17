import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import ica
from PIL import Image, ImageOps

import sys
sys.path.append('/data/users2/yxiao11/model/ICA')
from modules.util import entropyLoss, getdata, infomaxICA, entropy

file_name = os.listdir('./data')
dir_list = []
for name in file_name[0:5]:
    dir_list.append('/data/users2/yxiao11/model/ICA/data/'+name)
num_of_img = len(dir_list)
# num_of_img = 3
resize = 128

array_list=[]

for img_dir in dir_list:
    image = Image.open(img_dir)
    image = np.array(ImageOps.grayscale(image))
    image = cv2.resize(image, (resize, resize))
#     image = torch.from_numpy(image).float().flatten()
    image = np.expand_dims(image.flatten(), axis=0)
    array_list.append(image)

img_concate = np.concatenate(array_list)
np.random.seed(1)

# D = np.diag(2**np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=float))

# B = ortho_group.rvs(dim=num_of_img)
# A = B @ D
# A = np.random.normal(0, 1, size=(num_of_img,num_of_img))
A = np.random.rand(num_of_img,num_of_img)
mixture = A@img_concate

x_white, white, dewhite = ica.pca_whiten(mixture, num_of_img)

IPT = torch.from_numpy(x_white.copy()).type(torch.float32)


device = torch.device('cuda:0')
model = infomaxICA(num_of_img)


learning_rate = 0.003 / np.log(num_of_img)
batch_size = int(np.floor(np.sqrt(IPT.shape[1] / 3)))

dataset = getdata(IPT)
sampler = SequentialSampler(dataset)
loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, num_workers=3)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate,
                             eps=10e-4,
                             weight_decay = 1e-4,
                            )


num_epoch = 2000
scheduler = OneCycleLR(optimizer, 
                       learning_rate, 
                       steps_per_epoch=len(loader), 
                       epochs=num_epoch,
                      )

model = model.to(device)
IPT = IPT.to(device)
loss_tracker = []

def norm(x):
    """Computes the norm of a vector or the Frobenius norm of a
    matrix_rank
    """
    return torch.norm(torch.flatten(x))

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

#----training -----
k = int(batch_size*0.16)


# trace = []
change = 100
W_STOP = 1e-8
d_weigths = torch.zeros(num_of_img)
old_d_weights = torch.zeros([1, num_of_img])
torch_pi = torch.acos(torch.zeros(1)).item() * 2 

# for epoch in range(num_epoch):
step0 = 0
angle_delta = 0
while step0 < num_epoch and change > W_STOP:
# for i in range(1500):
    LOSS = 0
    old_weight = torch.clone(model.W1.weight.data)
    for step, ipt in enumerate(loader):
        model.zero_grad()
        ipt = ipt.to(device)
        
        opt = model.forward(ipt)
#         loss = entropy(opt, k=k, dis=1)
        loss = entropy(opt, k=k, dis=1) + 0.5*my_loss(opt)
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
    if step0 % 2 == 0:
        data1 = model.forward(IPT.T)
        data1 = data1.cpu().detach().numpy()
        # data = (IPT.T@model.W1.weight.data.T).cpu().detach().numpy()
#         index = np.random.randint(0,num_of_img, 2)
        index = [0,1]

    
        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1)
        plt.plot(loss_tracker[-100:])

        plt.subplot(1,3,2)
        plt.plot(loss_tracker)
        
        plt.subplot(1,3,3)
        plt.plot(data1[:, index[0]], data1[:, index[1]], '.', ms=0.5)
        
        # plt.subplot(1,3,3)
        # plt.plot(data[:, index[0]], data[:, index[1]], '.', ms=0.5)
        # fig.canvas.draw()
        
        plt.savefig('/data/users2/yxiao11/model/ICA/figures/images/experiments.png')
        plt.close()

    

    step0 += 1
    print(step0, angle_delta, LOSS.detach().numpy()/len(loader.sampler), change)
    scheduler.step()

#-------------
data = (IPT.T@model.W1.weight.data.T).cpu().detach().numpy()

plt.figure(figsize=(10,5*num_of_img))
for i in range(num_of_img):
    plt.subplot(num_of_img, 3, i*3+1)
    plt.imshow(np.reshape(array_list[i], (resize,resize)), cmap='gray')
    plt.title('original')
    plt.axis('off')
    
    plt.subplot(num_of_img, 3, i*3+2)
    plt.imshow(np.reshape(x_white[i], (resize,resize)), cmap='gray')
    plt.title('mixture')
    plt.axis('off')
    
    plt.subplot(num_of_img, 3, i*3+3)
    plt.imshow(np.reshape(data[:, i], (resize,resize)), cmap='gray')
    plt.title('output')
    plt.axis('off')
plt.savefig('/data/users2/yxiao11/model/ICA/figures/images/o_m_opt.png')
plt.close()


plt.figure(figsize=(10,10))
sub_shape = num_of_img
for i in range(sub_shape):
    for j in range(sub_shape):
        plt.subplot(sub_shape, sub_shape, i*sub_shape+j+1)
        plt.plot(data[:, i], data[:, j], '.', ms=1)
        plt.axis('square')
plt.savefig('/data/users2/yxiao11/model/ICA/figures/images/plot_compare.png')
plt.close()


mixer, b, unmixer = ica.ica1(mixture, ncomp=num_of_img, verbose=True)

map_k_i = np.corrcoef(data.T, b)
plt.imshow(abs(map_k_i), cmap='gist_heat')
plt.colorbar()
plt.savefig('/data/users2/yxiao11/model/ICA/figures/images/corr.png')
plt.close()

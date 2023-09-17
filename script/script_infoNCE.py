import torch
import numpy as np
import cv2
import os
import ica
import random

import torch.nn as nn
# from modules.util import entropy
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from modules.util import getdata, entropy

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
#-------------------------------------------------

file_name = os.listdir('./data')
dir_list = []
for name in file_name[0:6]:
    dir_list.append('./data/'+name)
num_of_img = len(dir_list)
resize = 128
array_list=[]
for img_dir in dir_list:
    image = Image.open(img_dir)
    image = np.array(ImageOps.grayscale(image))
    image = cv2.resize(image, (resize, resize))
    image = np.expand_dims(image.flatten(), axis=0)
    array_list.append(image)
img_concate = np.concatenate(array_list)
np.random.seed(1)
A = np.random.rand(num_of_img,num_of_img)
mixture = A@img_concate
# print(A, img_concate.shape, sep='\n')
# print('condition number:', np.linalg.cond(A))
#------------------------------------------------

class infomaxICA(nn.Module):

    def __init__(self, n, bias=True):
        super(infomaxICA, self).__init__()
        self.W1 = torch.nn.Linear(n, n, bias=bias)
#         with torch.no_grad():
#             self.W1.weight = nn.Parameter(torch.diag(torch.ones(n)))
#             self.W1.bias = nn.Parameter(torch.zeros(n))
        
        self.W_bn = torch.nn.BatchNorm1d(n, track_running_stats = False)
        for param in self.W_bn.parameters():
            param.requires_grad = False
            
            
#         self.encoder = nn.Sequential(
#             nn.Linear(n, n),
#             nn.ReLU(),
#             nn.Linear(n, n),
#             nn.ReLU(),
#             nn.Linear(n, 1),
#         )
        
#         self.W_bn1 = torch.nn.BatchNorm1d(1, track_running_stats = False)
#         for param in self.W_bn1.parameters():
#             param.requires_grad = False
        # self.weight = torch.nn.parameter.Parameter(torch.rand(1), requires_grad=True)
        self.init_weight()

    def weights_init(self, m, layer_type=nn.Linear):
        if isinstance(m, layer_type):
            nn.init.xavier_normal_(m.weight.data)

    def init_weight(self):
        for layer in [nn.Linear]:
            self.apply(lambda x: self.weights_init(x, layer_type=layer))

    def forward(self, input):
        output_w1 = self.W1(input)  
        output_w1 = self.W_bn(output_w1)
        
#         code = self.encoder(output_w1)
#         code = self.W_bn1(code)
        return output_w1
#--------------------------------------------------------------
class infoNCE(nn.Module):
    '''
    opt is in the shape of nn output (batchsize, d)
    '''
    
    def __init__(self, num_of_img, num_of_neg):
        super(infoNCE, self).__init__()
        self.q= num_of_neg//num_of_img
        self.num_of_neg = num_of_neg
        torch.manual_seed(0)
        self.W = torch.randn(self.q + 1, num_of_img, num_of_img)
        self.W = self.W.cuda()
        

    
    
    def forward(self, opt, ipt_batch):
        '''
        neg_batch is in the shape of (batchsize, d)
        '''
        batchsize, d = opt.shape
        
        #-------negative based on mixture----
#         neg_batch = []
#         for i in range(self.q+1):
#             neg_batch.append(ipt_batch@self.W[i])
#         neg_batch = torch.concatenate(neg_batch, axis=1)[:, :self.num_of_neg]
#         neg_batch = neg_batch.cuda()
        #-------------------------------
        neg_batch = torch.randn(batchsize, self.num_of_neg).cuda()
        
        
        Ln = 0
        for i in range(d):
            
            neg_opt_batch = torch.concatenate([opt[:, i:i+1], neg_batch], axis=1)
            Ln_sub = 0
            a = [torch.exp(torch.norm(ipt_batch.T@neg_opt_batch[:, k])/100) for k in range(self.num_of_neg+1)]
            denominator = torch.stack(a).sum() + 1e-12
            for j in range(self.num_of_neg+1):
                
                numerator = torch.exp(torch.norm(ipt_batch.T@neg_opt_batch[:,j])/100) + 1e-12
 
                Ln_sub += torch.log(numerator/denominator)
            Ln += Ln_sub/(self.num_of_neg+1)
    #         print(int(numerator), int(denominator))

        return -Ln/(d)
#-----------------------------------------------------------------------
def my_entropy(x, n, k=3):
    _, d = x.shape
    loss=0
    for i in range(n):
        index = random.sample(range(d),3)
        loss += entropy(x[:, index], k=k)
    return loss/n
#---------------------------------------------------------
device = torch.device('cuda:0')
x_white, white, dewhite = ica.pca_whiten(mixture, num_of_img)
IPT = torch.from_numpy(x_white.copy()).type(torch.float32)

model = infomaxICA(num_of_img)
learning_rate = 0.003 / np.log(num_of_img)
# batch_size = int(np.floor(np.sqrt(IPT.shape[1] / 3)))
batch_size = 512
dataset = getdata(IPT)
sampler = SequentialSampler(dataset)
loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, num_workers=3)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate,
                             eps=10e-4,
                             weight_decay = 1e-4,
                            )

num_epoch = 200
scheduler = CosineAnnealingLR(optimizer, T_max = num_epoch/2)
# scheduler = OneCycleLR(optimizer, 
#                        learning_rate, 
#                        steps_per_epoch=len(loader), 
#                        epochs=num_epoch,
#                       )
model = model.to(device)
IPT = IPT.to(device)
my_loss = infoNCE(num_of_img, num_of_neg=100)

loss_tracker = []
#------------------training------------------------------
def norm(x):
    """Computes the norm of a vector or the Frobenius norm of a
    matrix_rank
    """
    return torch.norm(torch.flatten(x))
# k = int(batch_size*0.16)
k=3

# trace = []
change = 100
W_STOP = 1e-8
d_weigths = torch.zeros(num_of_img)
old_d_weights = torch.zeros([1, num_of_img])
torch_pi = torch.acos(torch.zeros(1)).item() * 2 

# for epoch in range(num_epoch):
step0 = 0
angle_delta = 0

stream = torch.cuda.Stream()

while step0 < num_epoch and change > W_STOP:
# for i in range(1000):
    LOSS = 0
    old_weight = torch.clone(model.W1.weight.data)
    for step, ipt_batch in enumerate(loader):
        model.zero_grad()
        ipt_batch = ipt_batch.to(device)
        
        opt= model.forward(ipt_batch)

#         loss = entropyLoss(opt.permute(1,0), device=device)
#         loss1 = entropy(opt, k=k, dis=1)
        loss = my_loss(opt, ipt_batch) + my_entropy(opt, n=num_of_img)

        loss.backward()
#         infoloss.backward()
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
        data = (IPT.T@model.W1.weight.data.T).cpu().detach().numpy()

        # index = [0,1]
        index = np.random.randint(0,num_of_img, 2)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(loss_tracker[-100:])
        plt.subplot(1,2,2)
        plt.plot(data1[:, index[0]], data1[:, index[1]], '.', ms=0.5)
        plt.savefig('./figures/nn_opt.png')
        plt.close()

    if step0 % 5 == 0:
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
        plt.savefig('./figures/results.png')
        plt.close()
    

    step0 += 1
    print(step0, angle_delta.item(), LOSS.detach().numpy()/len(loader.sampler), change.item())
    scheduler.step()


#-------------------------------------
mixer, b, unmixer = ica.ica1(mixture, ncomp=num_of_img, verbose=True)
data = (IPT.T@model.W1.weight.data.T).cpu().detach().numpy()

map_k_i = np.corrcoef(data.T, b)
map_k_white = np.corrcoef(data.T, x_white)
map_i_white = np.corrcoef(b, x_white)

plt.figure(figsize=(10,30))

plt.subplot(3,1,1)
plt.imshow(abs(map_k_i), cmap='gist_heat')
plt.colorbar()
plt.title('k_i')
plt.subplot(3,1,2)
plt.imshow(map_k_white, cmap='gist_heat')
plt.colorbar()
plt.title('k_white')
plt.subplot(3,1,3)
plt.imshow(map_i_white, cmap='gist_heat')
plt.title('i_white')
plt.colorbar()

plt.savefig('./figures/heatmap.png')
plt.close()

plt.figure(figsize=(10,10))
sub_shape = num_of_img
for i in range(sub_shape):
    for j in range(sub_shape):
        plt.subplot(sub_shape, sub_shape, i*sub_shape+j+1)
        plt.plot(data[:, i], data[:, j], '.', ms=1)
        plt.axis('square')
plt.savefig('./figures/compare.png')
plt.close()
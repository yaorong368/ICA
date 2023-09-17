import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from torch.linalg import svd

# from scipy.spatial import distance



def entropyLoss(X, device):
    d = X.shape[0]
    n = X.shape[1]

    if torch.cuda.is_available():
        eigen = torch.eye(n, n).to(device)
    else:
        eigen = torch.eye(n, n)

    expression = torch.cdist(X.T, X.T, p=2)
    # expression =torch.cdist(X.T, X.T, p=12)
    distance_sum = torch.sum(torch.log(expression + eigen + 1e-12))
    entropy = -d/(n*(n-1)) * distance_sum
    # squared_norms = (X**2).sum(0).repeat(n, 1)
    # squared_norms_T = squared_norms.T
    # X_T = X.T
    # arg = squared_norms + squared_norms_T - 2 * torch.mm(X_T, X)
    # expression = torch.sum(torch.log(torch.abs(arg)+ eigen + 1e-12))/2

    # entropy = -d/(n*(n-1)) * expression

    return entropy


class getdata(Dataset):
    def __init__(self, mix):
        self.mix = mix

    def __getitem__(self, item):
        opt = self.mix[:, item]
        # opt.requires_grad_()
        return opt

    def __len__(self):
        return self.mix.shape[1]


class infomaxICA(nn.Module):

    def __init__(self, n, bias=True):
        super(infomaxICA, self).__init__()
        self.W1 = torch.nn.Linear(n, n, bias=bias)
        with torch.no_grad():
            self.W1.weight = nn.Parameter(torch.diag(torch.ones(n)))
            self.W1.bias = nn.Parameter(torch.zeros(n))
        

        # self.W2 = torch.nn.Linear(n, n, bias = bias)
        # with torch.no_grad():
        #     self.W2.weight = nn.Parameter(torch.diag(torch.ones(n)))
        #     self.W2.bias = nn.Parameter(torch.zeros(n))

        self.W_bn = torch.nn.BatchNorm1d(n, track_running_stats = False)
        for param in self.W_bn.parameters():
            param.requires_grad = False

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
        # output_w2 = self.W2(output_w1)
        # output_w2 = self.W_bn(output_w2)
        return output_w1
        # return torch.sigmoid(output)
        # return torch.pi*output
        # return output**3/3+output
        # return self.W_bn(output)
        # return torch.pi*torch.tanh(output)
        # return torch.pi*torch.sin(output)


def knn(x, y, k=3, last_only=False, discard_nearest=True, dis=1):
    """Find k_neighbors-nearest neighbor distances from y for each example in a minibatch x.
    :param x: tensor of shape [T, N]
    :param y: tensor of shape [T', N]
    :param k: the (k_neighbors+1):th nearest neighbor
    :param last_only: use only the last knn vs. all of them
    :param discard_nearest:
    :return: knn distances of shape [T, k_neighbors] or [T, 1] if last_only
    """

    if dis == 0:
        '''
        cosine similarity
        '''
        dot_p = torch.matmul(x, y.transpose(0,1))
        norm_x = torch.norm(x, dim=1).unsqueeze(1)
        norm_y = torch.norm(y, dim=1).unsqueeze(0)
        ab = torch.mm(norm_x, norm_y) + 1e-6
        distmat = 1 - dot_p/ab + 1e-6
    if dis == 1:
        dist_x = (x ** 2).sum(-1).unsqueeze(1)  # [T, 1]
        dist_y = (y ** 2).sum(-1).unsqueeze(0)  # [1, T']
        cross = - 2 * torch.mm(x, y.transpose(0, 1))  # [T, T']
        distmat = dist_x + cross + dist_y  # distance matrix between all points x, y
        distmat = torch.clamp(distmat, 1e-8, 1e+8)  # can have negatives otherwise!

    if discard_nearest:  # never use the shortest, since it can be the same point
        knn, _ = torch.topk(distmat, k + 1, largest=False)
        knn = knn[:, 1:]
    else:
        knn, _ = torch.topk(distmat, k, largest=False)

    if last_only:
        knn = knn[:, -1:]  # k_neighbors:th distance only

    return torch.sqrt(knn)


def kl_div(x, y, k=3, eps=1e-8, last_only=False):
    """KL divergence estimator for batches x~p(x), y~p(y).
    :param x: prediction; shape [T, N]
    :param y: target; shape [T', N]
    :param k:
    :return: scalar
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x.astype(np.float32))
        y = torch.tensor(y.astype(np.float32))

    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True)
    nns_xy = knn(x, y, k=k, last_only=last_only, discard_nearest=False)

    divergence = (torch.log(nns_xy + eps) - torch.log(nns_xx + eps)).mean()

    return divergence


def entropy(x, k=3, eps=1e-8, last_only=False, dis=1):
    """Entropy estimator for batch x~p(x).
        :param x: prediction; shape [T, N]
        :param k:
        :return: scalar
        """
    if type(x) is np.ndarray:
        x = torch.tensor(x.astype(np.float32))

    # x = (0.5+x)/2

    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True, dis=dis)

    ent = torch.log(nns_xx + eps).mean() - torch.log(torch.tensor(eps))

    return -ent


# def batch_loss(x, loop=3, k=3):
#     d = x.shape[1]
#     perm = torch.randperm(d)
#     loss = torch.tensor([0]).type(torch.float32).cuda()
#     for i in range(int(np.ceil(d/3))):
#         if len(perm)>3:
#             loss += entropy(opt[:, perm[0:3]], k=k)
#             perm = perm[3:]
#         else:
#             loss += entropy(opt[:, perm], k=k)
            
            
#     return loss/int(np.ceil(d/3))
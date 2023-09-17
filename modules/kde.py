import torch
import numpy as np

def kernel_density_estimator(X, X_test, kernel='gaussian', bandwidth=1.0):
    n = X.shape[0]
    n_test = X_test.shape[0]
    n_train = X.shape[0]

    # Compute pairwise distances between test and training data
    test_norms = torch.sum(X_test**2, dim=1).view(-1, 1)
    train_norms = torch.sum(X**2, dim=1).view(1, -1)
    dot_products = torch.mm(X_test, X.t())
    norms = test_norms - 2 * dot_products + train_norms
    norms /= bandwidth**2
    norms.clamp_(min=0)

    # Compute kernel values
    if kernel == 'gaussian':
        kernel_vals = torch.exp(-0.5 * norms) / np.sqrt(2 * np.pi)
    elif kernel == 'epanechnikov':
        kernel_vals = torch.where(norms < 1, 0.75 * (1 - norms), torch.zeros_like(norms))
    else:
        raise ValueError('Invalid kernel type.')

    # Compute log-density estimate
    log_densities = torch.log(torch.mean(kernel_vals, dim=1)) - torch.log(torch.tensor(n_train).float()) 
    
    return torch.sum(log_densities)/(n*bandwidth)

X_text = torch.rand(10000,num_of_img).to(device)
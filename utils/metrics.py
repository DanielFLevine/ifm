import logging

import numpy as np
import ot
import torch
import umap
from sklearn.metrics import pairwise
from scipy.spatial.distance import cdist

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def mmd_rbf(X, Y, gamma=2.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = pairwise.rbf_kernel(X, X, gamma)
    YY = pairwise.rbf_kernel(Y, Y, gamma)
    XY = pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def energy_distance(X, Y):
    """Compute the Energy Distance between two distributions X and Y.

    Parameters:
    X : numpy array of shape (n_samples_X, n_features)
    Y : numpy array of shape (n_samples_Y, n_features)

    Returns:
    energy_dist : float
    """
    # Pairwise distances within X
    XX_dist = cdist(X, X, metric='euclidean')
    # Pairwise distances within Y
    YY_dist = cdist(Y, Y, metric='euclidean')
    # Pairwise distances between X and Y
    XY_dist = cdist(X, Y, metric='euclidean')
    
    # Compute mean distances
    XX_mean = XX_dist.mean()
    YY_mean = YY_dist.mean()
    XY_mean = XY_dist.mean()

    # Energy distance formula
    energy_dist = 2 * XY_mean - XX_mean - YY_mean
    
    return energy_dist


def compute_wass(X, Y, reg=0.01):
    # Compute the cost matrix (squared Euclidean distances)
    logger.info("Computing distance matrix for 2-Wasserstein...")
    M = ot.dist(X, Y, metric='sqeuclidean')

    # Normalize the cost matrix
    M /= M.max()

    # Assume uniform distribution of weights
    a, b = np.ones((X.shape[0],)) / X.shape[0], np.ones((Y.shape[0],)) / Y.shape[0]

    logger.info("Computing 2-Wasserstein distance...")
    wasserstein_dist = ot.sinkhorn2(a, b, M, reg=reg)
    return wasserstein_dist

def transform_gpu(data, pca):
    components = pca.components_
    mean = pca.mean_

    data_gpu = torch.tensor(data, device='cuda')
    components_gpu = torch.tensor(components, device='cuda')
    mean_gpu = torch.tensor(mean, dtype=torch.float32).cuda()

    # Center the data by subtracting the mean
    data_centered = data_gpu - mean_gpu
    
    # Transform the data using the PCA components
    transformed_data_gpu = torch.matmul(data_centered, components_gpu.T)
    
    transformed_data = transformed_data_gpu.cpu().numpy()
    return transformed_data

def umap_embed(
    gt_data,
    gen_data,
):
    
    # Combine the two datasets
    combined_data = np.vstack((gt_data, gen_data))

    # Fit and transform the combined data using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(combined_data)

    # Split the transformed data back into the two original sets
    num_samples = gt_data.shape[0]
    umap_gt = umap_embedding[:num_samples]
    umap_gen = umap_embedding[num_samples:]

    return umap_gt, umap_gen

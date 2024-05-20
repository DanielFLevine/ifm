import argparse
import json
import logging
import math
import os
import pickle
import random
import time
from datetime import datetime
from itertools import cycle
from tqdm import tqdm

import anndata
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import safetensors
import torch
import torchdyn
from datasets import load_from_disk, Dataset
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise
from sklearn.decomposition import PCA
from torchdyn.core import NeuralODE
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM, GPTNeoXConfig

from scvi.model import SCVI
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

from utils.modules import MLPLayer, MidFC, CustomDecoder, CustomVAEDecoder

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionFC(nn.Module):
    def __init__(self, intermediate_dim=1024, num_fc_layers=2, denoising_time_steps=100):
        super(DiffusionFC, self).__init__()
        self.time_embed = nn.Embedding(denoising_time_steps, intermediate_dim)
        self.model = nn.Sequential(
            nn.Linear(768, intermediate_dim),
            MidFC(dim=intermediate_dim, num_layers=num_fc_layers),
            nn.Linear(intermediate_dim, 768)
        )
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x = self.model[0](x) + t_embed
        for layer in self.model[1:]:
            x = layer(x)
        return x


class DDPM:
    def __init__(self, model, device, num_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,)).to(self.device)

    def forward_diffusion(self, x_0, t):
        alpha_bars_t = self.alpha_bars[t].unsqueeze(1)
        print(alpha_bars_t)
        noise = torch.randn_like(x_0).to(self.device)
        x_t = torch.sqrt(alpha_bars_t) * x_0 + torch.sqrt(1 - alpha_bars_t) * noise
        return x_t, noise

    def denoise_step(self, x, t):
        alpha_bars_t = self.alpha_bars[t].unsqueeze(1)
        beta_t = self.betas[t].unsqueeze(1)
        alpha_t = self.alphas[t].unsqueeze(1)
        noise = torch.randn_like(x).to(self.device)
        added_noise = (torch.sqrt(beta_t)*noise)
        noise_pred = self.model(x, t)
        noise_pred_coeff = (1-alpha_t)/torch.sqrt(1 - alpha_bars_t)
        x_prev = (x - noise_pred_coeff * noise_pred) / torch.sqrt(alpha_t)
        x_prev = x_prev + added_noise
        return x_prev

    def denoise(self, x):
        x_t = x
        for t in range(self.num_timesteps-1, -1, -1):
            t_tensor = torch.tensor([t], dtype=torch.long).to(x.device)
            x_t = self.denoise_step(x_t, t_tensor)
        return x_t

    def loss(self, x_0):
        batch_size = x_0.shape[0]
        t = self.sample_timesteps(batch_size).to(x_0.device)
        x_t, noise = self.forward_diffusion(x_0, t)
        noise_pred = self.model(x_t, t)
        return nn.MSELoss()(noise_pred, noise)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=5
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000
    )
    parser.add_argument(
        "--gw",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--n_cell_thresh",
        type=int,
        default=100
    )
    parser.add_argument(
        "--hvgs",
        type=int,
        default=200
    )
    return parser.parse_args()

def mmd_rbf(X, Y, gamma=2.0, num_comps=50):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    X_red = X[:, :num_comps]
    Y_red = Y[:, :num_comps]
    XX = pairwise.rbf_kernel(X_red, X_red, gamma)
    YY = pairwise.rbf_kernel(Y_red, Y_red, gamma)
    XY = pairwise.rbf_kernel(X_red, Y_red, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def compute_wass(X, Y, reg=0.01, num_comps=50):
    X_red = X[:, :num_comps]
    Y_red = Y[:, :num_comps]
    # Compute the cost matrix (squared Euclidean distances)
    M = ot.dist(X_red, Y_red, metric='sqeuclidean')
    
    # Normalize the cost matrix
    M /= M.max()
    
    # Assume uniform distribution of weights
    a = np.ones((X_red.shape[0],)) / X_red.shape[0]
    b = np.ones((Y_red.shape[0],)) / Y_red.shape[0]
    
    wasserstein_dist = ot.sinkhorn2(a, b, M, reg)
    return wasserstein_dist

def compute_gw(X, Y, num_comps=50):
    X_red = X[:, :num_comps]
    Y_red = Y[:, :num_comps]
    C1 = ot.dist(X_red, X_red, metric='euclidean')
    C2 = ot.dist(Y_red, Y_red, metric='euclidean')

    # Uniform distributions over samples
    p = ot.unif(X_red.shape[0])
    q = ot.unif(Y_red.shape[0])

    # Compute Gromov-Wasserstein distance
    gw_dist = ot.gromov_wasserstein2(C1, C2, p, q, 'square_loss')
    return gw_dist

def compute_dists(X, Y, compute_gw=False):
    mmd = mmd_rbf_gpu(X, Y)
    wass = compute_wass(X, Y)
    if compute_gw:
        gw = compute_gw(X, Y)
    else:
        gw = 1.0
    return mmd, wass, gw


def compute_statistics(array1, array2):
    # Ensure the arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("The input arrays must have the same shape.")
    
    # Calculate row-wise means
    mean1 = np.mean(array1, axis=0)
    mean2 = np.mean(array2, axis=0)
    
    # Calculate R^2
    correlation_matrix = np.corrcoef(mean1, mean2)
    r2 = correlation_matrix[0, 1] ** 2
    
    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(mean1, mean2)
    
    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(mean1, mean2)
    
    return r2, pearson_corr, spearman_corr

def inverse_transform_gpu(data, pca_model):
    components = pca_model.components_
    mean = pca_model.mean_
    
    # Convert to torch tensors and move to GPU
    data_torch = torch.tensor(data, dtype=torch.float32).cuda()
    components_torch = torch.tensor(components, dtype=torch.float32).cuda()
    mean_torch = torch.tensor(mean, dtype=torch.float32).cuda()

    # Perform the inverse transform using matrix multiplication on the GPU
    data_inverse_torch = torch.matmul(data_torch, components_torch) + mean_torch
    data_inverse = data_inverse_torch.cpu().numpy()  # Move back to CPU and convert to numpy array
    
    return data_inverse

def rbf_kernel(X, Y, gamma):
    """Compute the RBF (Gaussian) kernel between two matrices X and Y."""
    # Compute the pairwise squared Euclidean distances
    dist = torch.cdist(X, Y, p=2).pow(2)
    # Compute the RBF kernel
    K = torch.exp(-gamma * dist)
    return K

def mmd_rbf_gpu(X, Y, gamma=2.0, num_comps=50):
    """
    Compute the Maximum Mean Discrepancy (MMD) with an RBF (Gaussian) kernel on GPU.

    Arguments:
    - X: np.ndarray of shape (n_sample1, dim)
    - Y: np.ndarray of shape (n_sample2, dim)
    - gamma: float, kernel parameter (default: 2.0)
    - num_comps: int, number of components to use (default: 50)

    Returns:
    - float, MMD value
    """
    # Convert to torch tensors and move to GPU
    X_torch = torch.tensor(X[:, :num_comps], dtype=torch.float32).cuda()
    Y_torch = torch.tensor(Y[:, :num_comps], dtype=torch.float32).cuda()

    # Compute the RBF kernels
    XX = rbf_kernel(X_torch, X_torch, gamma)
    YY = rbf_kernel(Y_torch, Y_torch, gamma)
    XY = rbf_kernel(X_torch, Y_torch, gamma)

    # Compute the MMD value
    mmd_value = XX.mean() + YY.mean() - 2 * XY.mean()
    
    return mmd_value.item()



def main(args):
    # Prep data

    adata = sc.read_h5ad("/home/dfl32/project/ifm/cinemaot_data/raw_cinemaot.h5ad")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=args.hvgs)

    hvgs = adata.var['highly_variable']
    rare_hvgs = hvgs & (adata.var['n_cells'] < args.n_cell_thresh)

    if issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X

    # Load saved PCA model
    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"pcadim768_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)

    # Load scaler for diffusion
    with open('/home/dfl32/project/ifm/scalers/pcadim768_numsamples2000_minmax.pickle', 'rb') as f:
        scaler = pickle.load(f)

    # Set number of repeats and samples
    num_repeats = args.num_repeats
    num_samples = args.num_samples

    # Set device
    device = torch.device("cuda")

    # Set loop parameters
    batch_size = 100
    num_steps = num_samples//batch_size

    ### IFM ###
    # Load IFM model
    model_name = "EleutherAI/pythia-160m"
    input_dim = 768
    config = GPTNeoXConfig(
            hidden_size=768,
            intermediate_size=1024,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=100,
            # use_flash_attention_2=args.use_flash_attention_2
            )
    model = GPTNeoXForCausalLM(config).to(device)
    model.cell_enc = nn.Linear(input_dim, model.config.hidden_size).to(device)
    model.cell_dec = CustomVAEDecoder(
        hidden_size=config.hidden_size,
        input_dim=input_dim,
        device=device,
        num_blocks=1
    )

    cp_dir = "/home/dfl32/scratch/training-runs/"
    run_name = "traincustomTrue-vaeTrue-klw0.3-EleutherAI/pythia-160m-timepoints16-straightpathTrue-drop0.0ifm-2024-05-16_16-12-17"
    cp_num = 60000
    cp = f"checkpoint-{cp_num}"
    model_weights_path = os.path.join(cp_dir, run_name, cp, "model.safetensors")
    pt_state_dict = safetensors.torch.load_file(model_weights_path, device="cuda")
    logger.info(model.load_state_dict(pt_state_dict))
    model.eval()

    ifm_mmds = []
    ifm_wasss = []
    ifm_gws = []
    ifm_mmds_rare = []
    ifm_wasss_rare = []
    ifm_gws_rare = []
    time_points = 16
    for _ in range(num_repeats):
        with torch.no_grad():
            cells = []
            for step in tqdm(range(num_steps)):
                inputs = torch.normal(0.0, 1.0, size=(batch_size, 1, input_dim)).to(device)
                for time in range(time_points-1):
                    outputs = model.cell_enc(inputs)
                    outputs = model.gpt_neox(inputs_embeds=outputs).last_hidden_state
                    outputs, _, _ = model.cell_dec(outputs, temperature=args.temp)
                    inputs = torch.concat([inputs, outputs[:, -1:, :]], axis=1)
                cells.append(outputs[:, -1, :].detach().cpu().numpy())
            cells = np.concatenate(cells, axis=0)

        logger.info("Inverse transforming IFM generated cells...")
        # cells_ag = pca.inverse_transform(cells)
        cells_ag = inverse_transform_gpu(cells, pca)
        logger.info("Done.")
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        # HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"IFM HVGS MMD: {mmd}")
        logger.info(f"IFM HVGS Wass: {wass}")
        logger.info(f"IFM HVGS GW: {gw}")
        ifm_mmds.append(mmd)
        ifm_wasss.append(wass)
        ifm_gws.append(gw)

        # Rare HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"IFM Rare HVGS MMD: {mmd}")
        logger.info(f"IFM Rare HVGS Wass: {wass}")
        logger.info(f"IFM Rare HVGS GW: {gw}")
        ifm_mmds_rare.append(mmd)
        ifm_wasss_rare.append(wass)
        ifm_gws_rare.append(gw)

    ifm_mmds = np.array(ifm_mmds)
    ifm_wasss = np.array(ifm_wasss)
    ifm_gws = np.array(ifm_gws)
    logger.info(f"IFM HVG MMD Mean {ifm_mmds.mean()} STD {ifm_mmds.std()}")
    logger.info(f"IFM HVG Wass Mean {ifm_wasss.mean()} STD {ifm_wasss.std()}")
    logger.info(f"IFM HVG GW Mean {ifm_gws.mean()} STD {ifm_gws.std()}")

    ifm_mmds_rare = np.array(ifm_mmds_rare)
    ifm_wasss_rare = np.array(ifm_wasss_rare)
    ifm_gws_rare = np.array(ifm_gws_rare)
    logger.info(f"IFM Rare HVG MMD Mean {ifm_mmds_rare.mean()} STD {ifm_mmds_rare.std()}")
    logger.info(f"IFM Rare HVG Wass Mean {ifm_wasss_rare.mean()} STD {ifm_wasss_rare.std()}")
    logger.info(f"IFM Rare HVG GW Mean {ifm_gws_rare.mean()} STD {ifm_gws_rare.std()}")



    ### CFM ###
    # Load CFM model
    model_dir = "/home/dfl32/scratch/training-runs/simple_ifm/cfm-mlp-2024-05-17_23-11-41/"
    checkpoint = 100000
    model_path = os.path.join(model_dir, f"checkpoint-{checkpoint}.pt")
    device = torch.device("cuda")
    input_dim = 768
    mlp_width = 1024

    model = MLP(
        dim=input_dim,
        w=mlp_width, 
        time_varying=True
    ).to(device)
    print(model.load_state_dict(torch.load(model_path)))
    model.eval()

    cfm_mmds = []
    cfm_wasss = []
    cfm_gws = []
    cfm_mmds_rare = []
    cfm_wasss_rare = []
    cfm_gws_rare = []
    node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    for _ in range(num_repeats):
        with torch.no_grad():
            cells = []
            for step in tqdm(range(num_steps)):
                x0 = torch.normal(0.0, 1.0**0.5, size=(batch_size, input_dim)).to(device)
                traj = node.trajectory(
                            x0,
                            t_span=torch.linspace(0, 1, 100),
                        ) # shape num_time_points x batch_size x output_dim
                cells.append(traj[-1, :, :].cpu().numpy())
            cells = np.concatenate(cells, axis=0)
        logger.info("Inverse transforming CFM generated cells...")
        # cells_ag = pca.inverse_transform(cells)
        cells_ag = inverse_transform_gpu(cells, pca)
        logger.info("Done.")
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        # HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"CFM HVGS MMD: {mmd}")
        logger.info(f"CFM HVGS Wass: {wass}")
        logger.info(f"CFM HVGS GW: {gw}")
        cfm_mmds.append(mmd)
        cfm_wasss.append(wass)
        cfm_gws.append(gw)

        # Rare HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"CFM Rare HVGS MMD: {mmd}")
        logger.info(f"CFM Rare HVGS Wass: {wass}")
        logger.info(f"CFM Rare HVGS GW: {gw}")
        cfm_mmds_rare.append(mmd)
        cfm_wasss_rare.append(wass)
        cfm_gws_rare.append(gw)

    cfm_mmds = np.array(cfm_mmds)
    cfm_wasss = np.array(cfm_wasss)
    cfm_gws = np.array(cfm_gws)
    logger.info(f"CFM HVG MMD Mean {cfm_mmds.mean()} STD {cfm_mmds.std()}")
    logger.info(f"CFM HVG Wass Mean {cfm_wasss.mean()} STD {cfm_wasss.std()}")
    logger.info(f"CFM HVG GW Mean {cfm_gws.mean()} STD {cfm_gws.std()}")

    cfm_mmds_rare = np.array(cfm_mmds_rare)
    cfm_wasss_rare = np.array(cfm_wasss_rare)
    cfm_gws_rare = np.array(cfm_gws_rare)
    logger.info(f"CFM Rare HVG MMD Mean {cfm_mmds_rare.mean()} STD {cfm_mmds_rare.std()}")
    logger.info(f"CFM Rare HVG Wass Mean {cfm_wasss_rare.mean()} STD {cfm_wasss_rare.std()}")
    logger.info(f"CFM Rare HVG GW Mean {cfm_gws_rare.mean()} STD {cfm_gws_rare.std()}")


    ### Diffusion ###
    # Load diffusion model
    model_dir = "/home/dfl32/scratch/training-runs/simple_ifm/diffusion-2024-05-19_13-31-26"
    checkpoint = 1000000
    model_path = os.path.join(model_dir, f"checkpoint-{checkpoint}.pt")
    device = torch.device("cuda")


    denoising_time_steps = 1000
    intermediate_dim = 2048
    num_fc_layers = 1
    model = DiffusionFC(
        intermediate_dim=intermediate_dim,
        denoising_time_steps=denoising_time_steps,
        num_fc_layers=num_fc_layers
    ).to(device)
    print(model.load_state_dict(torch.load(model_path)))
    model.eval()
    ddpm = DDPM(model, device, num_timesteps=denoising_time_steps)


    diff_mmds = []
    diff_wasss = []
    diff_gws = []
    diff_mmds_rare = []
    diff_wasss_rare = []
    diff_gws_rare = []
    for _ in range(num_repeats):
        cells = []
        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                x_noisy = torch.randn(batch_size, 768).to(device)  # Replace with your noisy input
                x_denoised = ddpm.denoise(x_noisy)
                x_denoised = scaler.inverse_transform(x_denoised.cpu().numpy())
                cells.append(x_denoised)
            cells = np.concatenate(cells, axis=0)
        
        logger.info("Inverse transforming diffusion generated cells...")
        # cells_ag = pca.inverse_transform(cells)
        cells_ag = inverse_transform_gpu(cells, pca)
        logger.info("Done.")
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        # HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"Diffusion HVGS MMD: {mmd}")
        logger.info(f"Diffusion HVGS Wass: {wass}")
        logger.info(f"Diffusion HVGS GW: {gw}")
        diff_mmds.append(mmd)
        diff_wasss.append(wass)
        diff_gws.append(gw)

        # Rare HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"Diffusion Rare HVGS MMD: {mmd}")
        logger.info(f"Diffusion Rare HVGS Wass: {wass}")
        logger.info(f"Diffusion Rare HVGS GW: {gw}")
        diff_mmds_rare.append(mmd)
        diff_wasss_rare.append(wass)
        diff_gws_rare.append(gw)

    diff_mmds = np.array(diff_mmds)
    diff_wasss = np.array(diff_wasss)
    diff_gws = np.array(diff_gws)
    logger.info(f"Diffusion HVG MMD Mean {diff_mmds.mean()} STD {diff_mmds.std()}")
    logger.info(f"Diffusion HVG Wass Mean {diff_wasss.mean()} STD {diff_wasss.std()}")
    logger.info(f"Diffusion HVG GW Mean {diff_gws.mean()} STD {diff_gws.std()}")

    diff_mmds_rare = np.array(diff_mmds_rare)
    diff_wasss_rare = np.array(diff_wasss_rare)
    diff_gws_rare = np.array(diff_gws_rare)
    logger.info(f"Diffusion Rare HVG MMD Mean {diff_mmds_rare.mean()} STD {diff_mmds_rare.std()}")
    logger.info(f"Diffusion Rare HVG Wass Mean {diff_wasss_rare.mean()} STD {diff_wasss_rare.std()}")
    logger.info(f"Diffusion Rare HVG GW Mean {diff_gws_rare.mean()} STD {diff_gws_rare.std()}")


    ### SCVI ###
    # Load model
    model = SCVI.load(
        dir_path="/home/dfl32/project/ifm/scvi_models/",
        prefix="epoch70_layers2_latent10_hidden128_pathsFalse",
        adata=adata)
    
    latent_dim = 10
    total_counts = adata.obs['total_counts']
    median_counts = total_counts.median()
    size_factors = total_counts / median_counts

    scvi_mmds = []
    scvi_wasss = []
    scvi_gws = []
    scvi_mmds_rare = []
    scvi_wasss_rare = []
    scvi_gws_rare = []
    for _ in range(num_repeats):
        cells = []
        with torch.no_grad():
            z_samples = torch.randn(num_samples, latent_dim)
            library_sizes = torch.tensor(np.random.choice(size_factors, num_samples)).unsqueeze(1)
            batch_index = torch.zeros(num_samples, dtype=torch.int64)
            generated_data = model.module.generative(z=z_samples, batch_index=batch_index, library=library_sizes)
            cells = generated_data['px'].sample().cpu().numpy()

        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        cells_ag = cells
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        # HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"SCVI HVGS MMD: {mmd}")
        logger.info(f"SCVI HVGS Wass: {wass}")
        logger.info(f"SCVI HVGS GW: {gw}")
        scvi_mmds.append(mmd)
        scvi_wasss.append(wass)
        scvi_gws.append(gw)

        # Rare HVGS
        mmd, wass, gw = compute_dists(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"SCVI Rare HVGS MMD: {mmd}")
        logger.info(f"SCVI Rare HVGS Wass: {wass}")
        logger.info(f"SCVI Rare HVGS GW: {gw}")
        scvi_mmds_rare.append(mmd)
        scvi_wasss_rare.append(wass)
        scvi_gws_rare.append(gw)

    scvi_mmds = np.array(scvi_mmds)
    scvi_wasss = np.array(scvi_wasss)
    scvi_gws = np.array(scvi_gws)
    logger.info(f"SCVI HVG MMD Mean {scvi_mmds.mean()} STD {scvi_mmds.std()}")
    logger.info(f"SCVI HVG Wass Mean {scvi_wasss.mean()} STD {scvi_wasss.std()}")
    logger.info(f"SCVI HVG GW Mean {scvi_gws.mean()} STD {scvi_gws.std()}")

    scvi_mmds_rare = np.array(scvi_mmds_rare)
    scvi_wasss_rare = np.array(scvi_wasss_rare)
    scvi_gws_rare = np.array(scvi_gws_rare)
    logger.info(f"SCVI HVG MMD Mean {scvi_mmds_rare.mean()} STD {scvi_mmds_rare.std()}")
    logger.info(f"SCVI HVG Wass Mean {scvi_wasss_rare.mean()} STD {scvi_wasss_rare.std()}")
    logger.info(f"SCVI HVG GW Mean {scvi_gws_rare.mean()} STD {scvi_gws_rare.std()}")


    # Collect all average scores at end of log
    logger.info("\n\n\n")

    logger.info(f"Num samples {args.num_samples}")
    logger.info(f"IFM Temp {args.temp}")
    logger.info(f"Num cells threshold for rare HVGS {args.n_cell_thresh}")
    logger.info(f"Num HVGS {args.hvgs}")

    logger.info(f"IFM HVG MMD Mean {ifm_mmds.mean()} STD {ifm_mmds.std()}")
    logger.info(f"IFM HVG Wass Mean {ifm_wasss.mean()} STD {ifm_wasss.std()}")
    logger.info(f"IFM HVG GW Mean {ifm_gws.mean()} STD {ifm_gws.std()}")
    logger.info(f"IFM Rare HVG MMD Mean {ifm_mmds_rare.mean()} STD {ifm_mmds_rare.std()}")
    logger.info(f"IFM Rare HVG Wass Mean {ifm_wasss_rare.mean()} STD {ifm_wasss_rare.std()}")
    logger.info(f"IFM Rare HVG GW Mean {ifm_gws_rare.mean()} STD {ifm_gws_rare.std()}")

    logger.info("\n")

    logger.info(f"CFM HVG MMD Mean {cfm_mmds.mean()} STD {cfm_mmds.std()}")
    logger.info(f"CFM HVG Wass Mean {cfm_wasss.mean()} STD {cfm_wasss.std()}")
    logger.info(f"CFM HVG GW Mean {cfm_gws.mean()} STD {cfm_gws.std()}")
    logger.info(f"CFM Rare HVG MMD Mean {cfm_mmds_rare.mean()} STD {cfm_mmds_rare.std()}")
    logger.info(f"CFM Rare HVG Wass Mean {cfm_wasss_rare.mean()} STD {cfm_wasss_rare.std()}")
    logger.info(f"CFM Rare HVG GW Mean {cfm_gws_rare.mean()} STD {cfm_gws_rare.std()}")

    logger.info("\n")

    logger.info(f"Diffusion HVG MMD Mean {diff_mmds.mean()} STD {diff_mmds.std()}")
    logger.info(f"Diffusion HVG Wass Mean {diff_wasss.mean()} STD {diff_wasss.std()}")
    logger.info(f"Diffusion HVG GW Mean {diff_gws.mean()} STD {diff_gws.std()}")
    logger.info(f"Diffusion Rare HVG MMD Mean {diff_mmds_rare.mean()} STD {diff_mmds_rare.std()}")
    logger.info(f"Diffusion Rare HVG Wass Mean {diff_wasss_rare.mean()} STD {diff_wasss_rare.std()}")
    logger.info(f"Diffusion Rare HVG GW Mean {diff_gws_rare.mean()} STD {diff_gws_rare.std()}")

    logger.info("\n")

    logger.info(f"SCVI HVG MMD Mean {scvi_mmds.mean()} STD {scvi_mmds.std()}")
    logger.info(f"SCVI HVG Wass Mean {scvi_wasss.mean()} STD {scvi_wasss.std()}")
    logger.info(f"SCVI HVG GW Mean {scvi_gws.mean()} STD {scvi_gws.std()}")
    logger.info(f"SCVI HVG MMD Mean {scvi_mmds_rare.mean()} STD {scvi_mmds_rare.std()}")
    logger.info(f"SCVI HVG Wass Mean {scvi_wasss_rare.mean()} STD {scvi_wasss_rare.std()}")
    logger.info(f"SCVI HVG GW Mean {scvi_gws_rare.mean()} STD {scvi_gws_rare.std()}")

    

    

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)

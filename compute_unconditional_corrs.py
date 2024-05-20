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
        "--n_cell_thresh",
        type=int,
        default=100
    )
    parser.add_argument(
        "--hvgs",
        type=int,
        default=200
    )
    parser.add_argument(
        "--full_cell",
        type=bool,
        default=False
    )
    return parser.parse_args()

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

    ifm_r2s = []
    ifm_pears = []
    ifm_spears = []
    ifm_r2s_hvg = []
    ifm_pears_hvg = []
    ifm_spears_hvg = []
    ifm_r2s_hvg_rare = []
    ifm_pears_hvg_rare = []
    ifm_spears_hvg_rare = []
    time_points = 16
    for _ in range(num_repeats):
        with torch.no_grad():
            cells = []
            for step in tqdm(range(num_steps)):
                inputs = torch.normal(0.0, 1.0, size=(batch_size, 1, input_dim)).to(device)
                for time in range(time_points-1):
                    outputs = model.cell_enc(inputs)
                    outputs = model.gpt_neox(inputs_embeds=outputs).last_hidden_state
                    outputs, _, _ = model.cell_dec(outputs, temperature=1.2)
                    inputs = torch.concat([inputs, outputs[:, -1:, :]], axis=1)
                cells.append(outputs[:, -1, :].detach().cpu().numpy())
            cells = np.concatenate(cells, axis=0)

        logger.info("Inverse transforming IFM generated cells...")
        cells_ag = inverse_transform_gpu(cells, pca)
        # cells_ag = pca.inverse_transform(cells)
        logger.info("Done.")
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        # All genes
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag, sampled_expression_data)
        logger.info(f"IFM R^2: {r2}")
        logger.info(f"IFM Pearson correlation: {pearson_corr}")
        logger.info(f"IFM Spearman correlation: {spearman_corr}")
        ifm_r2s.append(r2)
        ifm_pears.append(pearson_corr)
        ifm_spears.append(spearman_corr)

        # HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"IFM HVGS R^2: {r2}")
        logger.info(f"IFM HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"IFM HVGS Spearman correlation: {spearman_corr}")
        ifm_r2s_hvg.append(r2)
        ifm_pears_hvg.append(pearson_corr)
        ifm_spears_hvg.append(spearman_corr)

        # Rare HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"IFM Rare HVGS R^2: {r2}")
        logger.info(f"IFM Rare HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"IFM Rare HVGS Spearman correlation: {spearman_corr}")
        ifm_r2s_hvg_rare.append(r2)
        ifm_pears_hvg_rare.append(pearson_corr)
        ifm_spears_hvg_rare.append(spearman_corr)

    ifm_r2s = np.array(ifm_r2s)
    ifm_pears = np.array(ifm_pears)
    ifm_spears = np.array(ifm_spears)
    logger.info(f"IFM R^2 Mean {ifm_r2s.mean()} STD {ifm_r2s.std()}")
    logger.info(f"IFM Pearson Mean {ifm_pears.mean()} STD {ifm_pears.std()}")
    logger.info(f"IFM Spearman Mean {ifm_spears.mean()} STD {ifm_spears.std()}")

    ifm_r2s_hvg = np.array(ifm_r2s_hvg)
    ifm_pears_hvg = np.array(ifm_pears_hvg)
    ifm_spears_hvg = np.array(ifm_spears_hvg)
    logger.info(f"IFM HVGS R^2 Mean {ifm_r2s_hvg.mean()} STD {ifm_r2s_hvg.std()}")
    logger.info(f"IFM HVGS Pearson Mean {ifm_pears_hvg.mean()} STD {ifm_pears_hvg.std()}")
    logger.info(f"IFM HVGS Spearman Mean {ifm_spears_hvg.mean()} STD {ifm_spears_hvg.std()}")

    ifm_r2s_hvg_rare = np.array(ifm_r2s_hvg_rare)
    ifm_pears_hvg_rare = np.array(ifm_pears_hvg_rare)
    ifm_spears_hvg_rare = np.array(ifm_spears_hvg_rare)
    logger.info(f"IFM Rare HVGS R^2 Mean {ifm_r2s_hvg_rare.mean()} STD {ifm_r2s_hvg_rare.std()}")
    logger.info(f"IFM Rare HVGS Pearson Mean {ifm_pears_hvg_rare.mean()} STD {ifm_pears_hvg_rare.std()}")
    logger.info(f"IFM Rare HVGS Spearman Mean {ifm_spears_hvg_rare.mean()} STD {ifm_spears_hvg_rare.std()}")



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

    cfm_r2s = []
    cfm_pears = []
    cfm_spears = []
    cfm_r2s_hvg = []
    cfm_pears_hvg = []
    cfm_spears_hvg = []
    cfm_r2s_hvg_rare = []
    cfm_pears_hvg_rare = []
    cfm_spears_hvg_rare = []
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
        cells_ag = inverse_transform_gpu(cells, pca)
        # cells_ag = pca.inverse_transform(cells)
        logger.info("Done.")
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        # All genes
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag, sampled_expression_data)
        logger.info(f"CFM R^2: {r2}")
        logger.info(f"CFM Pearson correlation: {pearson_corr}")
        logger.info(f"CFM Spearman correlation: {spearman_corr}")
        cfm_r2s.append(r2)
        cfm_pears.append(pearson_corr)
        cfm_spears.append(spearman_corr)

        # HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"CFM HVGS R^2: {r2}")
        logger.info(f"CFM HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"CFM HVGS Spearman correlation: {spearman_corr}")
        cfm_r2s_hvg.append(r2)
        cfm_pears_hvg.append(pearson_corr)
        cfm_spears_hvg.append(spearman_corr)

        # Rare HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"CFM Rare HVGS R^2: {r2}")
        logger.info(f"CFM Rare HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"CFM Rare HVGS Spearman correlation: {spearman_corr}")
        cfm_r2s_hvg_rare.append(r2)
        cfm_pears_hvg_rare.append(pearson_corr)
        cfm_spears_hvg_rare.append(spearman_corr)

    cfm_r2s = np.array(cfm_r2s)
    cfm_pears = np.array(cfm_pears)
    cfm_spears = np.array(cfm_spears)
    logger.info(f"CFM R^2 Mean {cfm_r2s.mean()} STD {cfm_r2s.std()}")
    logger.info(f"CFM Pearson Mean {cfm_pears.mean()} STD {cfm_pears.std()}")
    logger.info(f"CFM Spearman Mean {cfm_spears.mean()} STD {cfm_spears.std()}")

    cfm_r2s_hvg = np.array(cfm_r2s_hvg)
    cfm_pears_hvg = np.array(cfm_pears_hvg)
    cfm_spears_hvg = np.array(cfm_spears_hvg)
    logger.info(f"CFM HVGS R^2 Mean {cfm_r2s_hvg.mean()} STD {cfm_r2s_hvg.std()}")
    logger.info(f"CFM HVGS Pearson Mean {cfm_pears_hvg.mean()} STD {cfm_pears_hvg.std()}")
    logger.info(f"CFM HVGS Spearman Mean {cfm_spears_hvg.mean()} STD {cfm_spears_hvg.std()}")

    cfm_r2s_hvg_rare = np.array(cfm_r2s_hvg_rare)
    cfm_pears_hvg_rare = np.array(cfm_pears_hvg_rare)
    cfm_spears_hvg_rare = np.array(cfm_spears_hvg_rare)
    logger.info(f"CFM Rare HVGS R^2 Mean {cfm_r2s_hvg_rare.mean()} STD {cfm_r2s_hvg_rare.std()}")
    logger.info(f"CFM Rare HVGS Pearson Mean {cfm_pears_hvg_rare.mean()} STD {cfm_pears_hvg_rare.std()}")
    logger.info(f"CFM Rare HVGS Spearman Mean {cfm_spears_hvg_rare.mean()} STD {cfm_spears_hvg_rare.std()}")


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


    diff_r2s = []
    diff_pears = []
    diff_spears = []
    diff_r2s_hvg = []
    diff_pears_hvg = []
    diff_spears_hvg = []
    diff_r2s_hvg_rare = []
    diff_pears_hvg_rare = []
    diff_spears_hvg_rare = []
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
        cells_ag = inverse_transform_gpu(cells, pca)
        # cells_ag = pca.inverse_transform(cells)
        logger.info("Done.")
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        # All genes
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag, sampled_expression_data)
        logger.info(f"Diffusion R^2: {r2}")
        logger.info(f"Diffusion Pearson correlation: {pearson_corr}")
        logger.info(f"Diffusion Spearman correlation: {spearman_corr}")
        diff_r2s.append(r2)
        diff_pears.append(pearson_corr)
        diff_spears.append(spearman_corr)

        # HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"Diffusion HVGS R^2: {r2}")
        logger.info(f"Diffusion HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"Diffusion HVGS Spearman correlation: {spearman_corr}")
        diff_r2s_hvg.append(r2)
        diff_pears_hvg.append(pearson_corr)
        diff_spears_hvg.append(spearman_corr)

        # Rare HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"Diffusion Rare HVGS R^2: {r2}")
        logger.info(f"Diffusion Rare HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"Diffusion Rare HVGS Spearman correlation: {spearman_corr}")
        diff_r2s_hvg_rare.append(r2)
        diff_pears_hvg_rare.append(pearson_corr)
        diff_spears_hvg_rare.append(spearman_corr)

    diff_r2s = np.array(diff_r2s)
    diff_pears = np.array(diff_pears)
    diff_spears = np.array(diff_spears)
    logger.info(f"Diffusion R^2 Mean {diff_r2s.mean()} STD {diff_r2s.std()}")
    logger.info(f"Diffusion Pearson Mean {diff_pears.mean()} STD {diff_pears.std()}")
    logger.info(f"Diffusion Spearman Mean {diff_spears.mean()} STD {diff_spears.std()}")

    diff_r2s_hvg = np.array(diff_r2s_hvg)
    diff_pears_hvg = np.array(diff_pears_hvg)
    diff_spears_hvg = np.array(diff_spears_hvg)
    logger.info(f"Diffusion HVGS R^2 Mean {diff_r2s_hvg.mean()} STD {diff_r2s_hvg.std()}")
    logger.info(f"Diffusion HVGS Pearson Mean {diff_pears_hvg.mean()} STD {diff_pears_hvg.std()}")
    logger.info(f"Diffusion HVGS Spearman Mean {diff_spears_hvg.mean()} STD {diff_spears_hvg.std()}")

    diff_r2s_hvg_rare = np.array(diff_r2s_hvg_rare)
    diff_pears_hvg_rare = np.array(diff_pears_hvg_rare)
    diff_spears_hvg_rare = np.array(diff_spears_hvg_rare)
    logger.info(f"Diffusion Rare HVGS R^2 Mean {diff_r2s_hvg_rare.mean()} STD {diff_r2s_hvg_rare.std()}")
    logger.info(f"Diffusion Rare HVGS Pearson Mean {diff_pears_hvg_rare.mean()} STD {diff_pears_hvg_rare.std()}")
    logger.info(f"Diffusion Rare HVGS Spearman Mean {diff_spears_hvg_rare.mean()} STD {diff_spears_hvg_rare.std()}")


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

    scvi_r2s = []
    scvi_pears = []
    scvi_spears = []
    scvi_r2s_hvg = []
    scvi_pears_hvg = []
    scvi_spears_hvg = []
    scvi_r2s_hvg_rare = []
    scvi_pears_hvg_rare = []
    scvi_spears_hvg_rare = []
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

        # All genes
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag, sampled_expression_data)
        logger.info(f"SCVI R^2: {r2}")
        logger.info(f"SCVI Pearson correlation: {pearson_corr}")
        logger.info(f"SCVI Spearman correlation: {spearman_corr}")
        scvi_r2s.append(r2)
        scvi_pears.append(pearson_corr)
        scvi_spears.append(spearman_corr)

        # HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"SCVI HVGS R^2: {r2}")
        logger.info(f"SCVI HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"SCVI HVGS Spearman correlation: {spearman_corr}")
        scvi_r2s_hvg.append(r2)
        scvi_pears_hvg.append(pearson_corr)
        scvi_spears_hvg.append(spearman_corr)

        # Rare HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
        logger.info(f"SCVI Rare HVGS R^2: {r2}")
        logger.info(f"SCVI Rare HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"SCVI Rare HVGS Spearman correlation: {spearman_corr}")
        scvi_r2s_hvg_rare.append(r2)
        scvi_pears_hvg_rare.append(pearson_corr)
        scvi_spears_hvg_rare.append(spearman_corr)

    scvi_r2s = np.array(scvi_r2s)
    scvi_pears = np.array(scvi_pears)
    scvi_spears = np.array(scvi_spears)
    logger.info(f"SCVI R^2 Mean {scvi_r2s.mean()} STD {scvi_r2s.std()}")
    logger.info(f"SCVI Pearson Mean {scvi_pears.mean()} STD {scvi_pears.std()}")
    logger.info(f"SCVI Spearman Mean {scvi_spears.mean()} STD {scvi_spears.std()}")

    scvi_r2s_hvg = np.array(scvi_r2s_hvg)
    scvi_pears_hvg = np.array(scvi_pears_hvg)
    scvi_spears_hvg = np.array(scvi_spears_hvg)
    logger.info(f"SCVI HVGS R^2 Mean {scvi_r2s_hvg.mean()} STD {scvi_r2s_hvg.std()}")
    logger.info(f"SCVI HVGS Pearson Mean {scvi_pears_hvg.mean()} STD {scvi_pears_hvg.std()}")
    logger.info(f"SCVI HVGS Spearman Mean {scvi_spears_hvg.mean()} STD {scvi_spears_hvg.std()}")

    scvi_r2s_hvg_rare = np.array(scvi_r2s_hvg_rare)
    scvi_pears_hvg_rare = np.array(scvi_pears_hvg_rare)
    scvi_spears_hvg_rare = np.array(scvi_spears_hvg_rare)
    logger.info(f"SCVI Rare HVGS R^2 Mean {scvi_r2s_hvg_rare.mean()} STD {scvi_r2s_hvg_rare.std()}")
    logger.info(f"SCVI Rare HVGS Pearson Mean {scvi_pears_hvg_rare.mean()} STD {scvi_pears_hvg_rare.std()}")
    logger.info(f"SCVI Rare HVGS Spearman Mean {scvi_spears_hvg_rare.mean()} STD {scvi_spears_hvg_rare.std()}")


    # Collect all average scores at end of log
    logger.info(f"Num samples {args.num_samples}")
    logger.info(f"Num cells threshold for rare HVGS {args.n_cell_thresh}")
    logger.info(f"Num HVGS {args.hvgs}")
    logger.info("\n\n\n")

    logger.info(f"IFM R^2 Mean {ifm_r2s.mean()} STD {ifm_r2s.std()}")
    logger.info(f"IFM Pearson Mean {ifm_pears.mean()} STD {ifm_pears.std()}")
    logger.info(f"IFM Spearman Mean {ifm_spears.mean()} STD {ifm_spears.std()}")
    logger.info(f"IFM HVGS R^2 Mean {ifm_r2s_hvg.mean()} STD {ifm_r2s_hvg.std()}")
    logger.info(f"IFM HVGS Pearson Mean {ifm_pears_hvg.mean()} STD {ifm_pears_hvg.std()}")
    logger.info(f"IFM HVGS Spearman Mean {ifm_spears_hvg.mean()} STD {ifm_spears_hvg.std()}")
    logger.info(f"IFM Rare HVGS R^2 Mean {ifm_r2s_hvg_rare.mean()} STD {ifm_r2s_hvg_rare.std()}")
    logger.info(f"IFM Rare HVGS Pearson Mean {ifm_pears_hvg_rare.mean()} STD {ifm_pears_hvg_rare.std()}")
    logger.info(f"IFM Rare HVGS Spearman Mean {ifm_spears_hvg_rare.mean()} STD {ifm_spears_hvg_rare.std()}")

    logger.info("\n")

    logger.info(f"CFM R^2 Mean {cfm_r2s.mean()} STD {cfm_r2s.std()}")
    logger.info(f"CFM Pearson Mean {cfm_pears.mean()} STD {cfm_pears.std()}")
    logger.info(f"CFM Spearman Mean {cfm_spears.mean()} STD {cfm_spears.std()}")
    logger.info(f"CFM HVGS R^2 Mean {cfm_r2s_hvg.mean()} STD {cfm_r2s_hvg.std()}")
    logger.info(f"CFM HVGS Pearson Mean {cfm_pears_hvg.mean()} STD {cfm_pears_hvg.std()}")
    logger.info(f"CFM HVGS Spearman Mean {cfm_spears_hvg.mean()} STD {cfm_spears_hvg.std()}")
    logger.info(f"CFM Rare HVGS R^2 Mean {cfm_r2s_hvg_rare.mean()} STD {cfm_r2s_hvg_rare.std()}")
    logger.info(f"CFM Rare HVGS Pearson Mean {cfm_pears_hvg_rare.mean()} STD {cfm_pears_hvg_rare.std()}")
    logger.info(f"CFM Rare HVGS Spearman Mean {cfm_spears_hvg_rare.mean()} STD {cfm_spears_hvg_rare.std()}")

    logger.info("\n")

    logger.info(f"Diffusion R^2 Mean {diff_r2s.mean()} STD {diff_r2s.std()}")
    logger.info(f"Diffusion Pearson Mean {diff_pears.mean()} STD {diff_pears.std()}")
    logger.info(f"Diffusion Spearman Mean {diff_spears.mean()} STD {diff_spears.std()}")
    logger.info(f"Diffusion HVGS R^2 Mean {diff_r2s_hvg.mean()} STD {diff_r2s_hvg.std()}")
    logger.info(f"Diffusion HVGS Pearson Mean {diff_pears_hvg.mean()} STD {diff_pears_hvg.std()}")
    logger.info(f"Diffusion HVGS Spearman Mean {diff_spears_hvg.mean()} STD {diff_spears_hvg.std()}")
    logger.info(f"Diffusion Rare HVGS R^2 Mean {diff_r2s_hvg_rare.mean()} STD {diff_r2s_hvg_rare.std()}")
    logger.info(f"Diffusion Rare HVGS Pearson Mean {diff_pears_hvg_rare.mean()} STD {diff_pears_hvg_rare.std()}")
    logger.info(f"Diffusion Rare HVGS Spearman Mean {diff_spears_hvg_rare.mean()} STD {diff_spears_hvg_rare.std()}")

    logger.info("\n")

    logger.info(f"SCVI R^2 Mean {scvi_r2s.mean()} STD {scvi_r2s.std()}")
    logger.info(f"SCVI Pearson Mean {scvi_pears.mean()} STD {scvi_pears.std()}")
    logger.info(f"SCVI Spearman Mean {scvi_spears.mean()} STD {scvi_spears.std()}")
    logger.info(f"SCVI HVGS R^2 Mean {scvi_r2s_hvg.mean()} STD {scvi_r2s_hvg.std()}")
    logger.info(f"SCVI HVGS Pearson Mean {scvi_pears_hvg.mean()} STD {scvi_pears_hvg.std()}")
    logger.info(f"SCVI HVGS Spearman Mean {scvi_spears_hvg.mean()} STD {scvi_spears_hvg.std()}")
    logger.info(f"SCVI Rare HVGS R^2 Mean {scvi_r2s_hvg_rare.mean()} STD {scvi_r2s_hvg_rare.std()}")
    logger.info(f"SCVI Rare HVGS Pearson Mean {scvi_pears_hvg_rare.mean()} STD {scvi_pears_hvg_rare.std()}")
    logger.info(f"SCVI Rare HVGS Spearman Mean {scvi_spears_hvg_rare.mean()} STD {scvi_spears_hvg_rare.std()}")

    

    

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)

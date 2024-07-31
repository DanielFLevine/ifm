import argparse
import logging
import os
import pickle
from tqdm import tqdm

import numpy as np
import scanpy as sc
import torch
from torch import nn
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

from utils.modules import MidFC
from utils.metrics import mmd_rbf, compute_wass, transform_gpu

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionFC(nn.Module):
    def __init__(self, input_dim=1000, intermediate_dim=1024, num_fc_layers=2, denoising_time_steps=100):
        super(DiffusionFC, self).__init__()
        self.time_embed = nn.Embedding(denoising_time_steps, intermediate_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            MidFC(dim=intermediate_dim, num_layers=num_fc_layers),
            nn.Linear(intermediate_dim, input_dim)
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
        "--cp_dir",
        type=str,
        default="/home/dfl32/scratch/training-runs/"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=5
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1000
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
    save_name = f"pcadim{args.input_dim}_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)

    # Load scaler for diffusion
    with open('/home/dfl32/project/ifm/scalers/train_pcadim1000_minmax.pickle', 'rb') as f:
        scaler = pickle.load(f)

    # Set number of repeats and samples
    num_repeats = args.num_repeats
    num_samples = args.num_samples

    # Set device
    device = torch.device("cuda")

    # Set loop parameters
    batch_size = 100
    num_steps = num_samples//batch_size

     ### Diffusion ###
    # Load diffusion model
    model_path = os.path.join(args.cp_dir, f"checkpoint-{args.checkpoint}.pt")


    denoising_time_steps = 1000
    intermediate_dim = 2048
    num_fc_layers = 2
    model = DiffusionFC(
        input_dim=args.input_dim,
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
    mmds = []
    wasss = []
    for _ in range(num_repeats):
        cells = []
        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                x_noisy = torch.randn(batch_size, args.input_dim).to(device)  # Replace with your noisy input
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

        logger.info("PCAing ground truth data...")
        pca_sampled_expression_data = transform_gpu(sampled_expression_data, pca)
        logger.info("Done.")

        mmd = mmd_rbf(cells, pca_sampled_expression_data)
        mmds.append(mmd)
        logger.info(f"MMD: {mmd}")
        wass = compute_wass(cells, pca_sampled_expression_data)
        wasss.append(wass)
        logger.info(f"Wass: {wass}")

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
    mmds = np.array(mmds)
    wasss = np.array(wasss)
    logger.info(f"MMD Mean {mmds.mean()} STD {mmds.std()}")
    logger.info(f"2-Wasserstein Mean {wasss.mean()} STD {wasss.std()}\n")
    
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

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
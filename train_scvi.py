import argparse
import json
import logging
import os
import random
from collections import Counter

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from anndata import AnnData
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.combo_split import combo_split_nochron

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get random control expression for a given cell type
def get_random_ctr_expr(cell_type, cell_type_to_indices):
    idxs = cell_type_to_indices[cell_type]
    random_idx = np.random.choice(idxs)
    return adata_ctr.X[random_idx].toarray() if hasattr(adata_ctr.X, "toarray") else adata_ctr.X[random_idx]

def generate_paths(X_0, X_1, sigma=0.1, time_points=16):
    # Convert lists to tensors
    X_0 = torch.tensor(X_0, dtype=torch.float32).unsqueeze(1)
    X_1 = torch.tensor(X_1, dtype=torch.float32).unsqueeze(1)
    
    # Dimensions
    dim = X_0.shape[-1] # Here dim is 5000
    
    # Generate time points: from 0 to 1, including both endpoints, evenly spaced
    times = torch.linspace(0, 1, steps=time_points).view(time_points, 1)
    
    # Expand times and inputs to broadcast across dimensions
    times_expanded = times.expand(X_0.shape[0], time_points, dim)
    
    # Linear interpolation: tX_1 + (1-t)X_0 = X_0 + t(X_1 - X_0)
    path_means = X_0 + times_expanded * (X_1 - X_0)
    
    # Initialize paths with means (ensures exact start at X_0 and end at X_1)
    paths = path_means.clone()
    
    # Gaussian noise: zero mean, sigma standard deviation, but not for the first and last time points
    if time_points > 2:
        noise = sigma * torch.randn(time_points-2, dim)
        
        # Determine where X_0 or X_1 is non-zero, for intermediate time points
        non_zero_mask = ((X_0 != 0) | (X_1 != 0))
        non_zero_mask_expanded = non_zero_mask.unsqueeze(0).expand(time_points-2, -1)
        
        # Apply noise only where non_zero_mask is True, and only to intermediate points
        paths[1:-1] = paths[1:-1].where(~non_zero_mask_expanded, paths[1:-1] + noise)

    return paths.numpy()


def main(args):
    train_path = args.adata_path

    logger.info("Loading data...")
    adata = sc.read_h5ad(train_path)

    if args.raw_data:
        logger.info("Preprocessing adata...")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata = adata[adata.obs.n_genes_by_counts < 2500, :]
        adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if args.holdout_perts:
        test_combos = combo_split_nochron()
        adata.obs['cell_type_perturbation'] = list(zip(adata.obs['cell_type'], adata.obs['perturbation']))
        train_adata = adata[~adata.obs['cell_type_perturbation'].isin(test_combos)].copy()
        train_adata.obs.drop(columns=['cell_type_perturbation'], inplace=True)
    else:
        train_adata = adata


    # if args.paths:
    #     all_intermediates = []
    #     adata_pert = train_adata[train_adata.obs['perturbation'] != "No stimulation"].copy()
    #     adata_ctr = train_adata[train_adata.obs['perturbation'] == "No stimulation"].copy()
    #     cell_types = list(train_adata.obs['cell_type'].unique())
    #     perturbations = list(train_adata.obs['perturbation'].unique())
    #     for cell_type in cell_types:
    #         adata_ctr_filtered = adata_ctr[adata_ctr.obs['cell_type'] == cell_type].copy()
    #         adata_pert_filtered = adata_pert[adata_pert.obs['cell_type'] == cell_type].copy()

    #         for perturbation in perturbations:
    #             # Filter adata_pert for the current perturbation
    #             adata_pert_specific = adata_pert_filtered[adata_pert_filtered.obs['perturbation'] == perturbation].copy()
                
    #             # Determine the number of samples to match
    #             num_samples = adata_pert_specific.shape[0]
                
    #             # Check if we need to sample with or without replacement
    #             if num_samples > adata_ctr_filtered.shape[0]:
    #                 # Sample with replacement
    #                 sampled_indices = np.random.choice(adata_ctr_filtered.shape[0], num_samples, replace=True)
    #             else:
    #                 # Sample without replacement
    #                 sampled_indices = np.random.choice(adata_ctr_filtered.shape[0], num_samples, replace=False)
                
    #             # Get the sampled adata_ctr entries
    #             adata_ctr_sampled_np = adata_ctr_filtered[sampled_indices, :].X
    #             adata_pert_np = adata_pert_specific.X
    #             paths = generate_paths(adata_ctr_sampled_np, adata_pert_np)
    #             all_intermediates.append(paths[])

            

    logger.info("Setting up anndata...")
    scvi.model.SCVI.setup_anndata(
        train_adata)


    num_epochs = args.num_epochs
    save_dir = "/home/dfl32/project/ifm/scvi_models"

    n_layers = args.n_layers
    n_hidden = args.n_hidden
    n_latent = args.n_latent
    logger.info("Instantiating model...")
    model = scvi.model.SCVI(
        train_adata,
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_latent=n_latent
    )
    model.to_device("cuda")
    for epoch in range(num_epochs):
        logger.info(f"EPOCH {epoch}")
        model.train(
            accelerator='gpu', 
            batch_size=128,
            train_size=0.9,
            max_epochs=1,
            check_val_every_n_epoch=1
        )
        prefix = f"epoch{epoch}_layers{n_layers}_latent{n_latent}_hidden{n_hidden}_paths{args.paths}"
        model.save(save_dir, prefix=prefix, overwrite=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adata_path",
        type=str,
        default="/home/dfl32/project/ifm/cinemaot_data/raw_cinemaot.h5ad",
        help="Path to adata file for training.",
    )
    parser.add_argument(
        "--holdout_perts",
        type=bool,
        default=False,
        help="Hold out test perturbations.",
    )
    parser.add_argument(
        "--raw_data",
        type=bool,
        default=True,
        help="If data is raw counts.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers in scVI model.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=128,
        help="Hidden dimension in scVI model.",
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=10,
        help="Latent dimension scVI model.",
    )
    parser.add_argument(
        "--paths",
        type=bool,
        default=False,
        help="Whether to include interpolated paths.",
    )
    args = parser.parse_args()
    logger.info(args)
    main(args)
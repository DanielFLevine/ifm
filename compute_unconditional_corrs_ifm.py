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
from datasets import load_from_disk, Dataset
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM, GPTNeoXConfig


from utils.modules import CustomDecoder, CustomVAEDecoder

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)




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
        "--temp",
        type=float,
        default=0.7
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


def main(args):
    # Prep data

    adata = sc.read_h5ad("/home/dfl32/project/ifm/cinemaot_data/raw_cinemaot.h5ad")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=20)

    hvgs = adata.var['highly_variable']

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
        cells_ag = pca.inverse_transform(cells)
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

    logger.info(f"\n\n\nTEMP {args.temp}")
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


    

    

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)

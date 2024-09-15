import argparse
import logging
import os
import pickle

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

from scvi.model import SCVI

from utils.metrics import mmd_rbf, compute_wass, transform_gpu, umap_embed, evaluate_model
from utils.plots import plot_umap

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
        "--num_pca_dims",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--plot_umap",
        action="store_true",
    )
    parser.add_argument(
        "--umap_embed",
        action="store_true",
    )
    parser.add_argument(
        "--mmd_gamma",
        type=float,
        default=2.0
    )
    parser.add_argument(
        "--wass_reg",
        type=float,
        default=0.1
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


def add_gaussian_noise(array):
    noise = np.random.normal(loc=0, scale=1e-6, size=array.shape)
    noisy_array = array + noise
    return noisy_array

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
    print(sum(rare_hvgs))

    if issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X


    # Set number of repeats and samples
    num_repeats = args.num_repeats
    num_samples = args.num_samples

    # Load saved PCA model
    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"pcadim{args.input_dim}_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)


    ### SCVI ###
    # Load model
    model = SCVI.load(
    dir_path="/home/dfl32/project/ifm/scvi_models/",
    prefix="epoch99_layers2_latent10_hidden128_calmflow",
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
    mmds = []
    wasss = []
    avg_entropies = []
    kl_divs = []
    for i in range(num_repeats):
        cells = []
        with torch.no_grad():
            z_samples = torch.randn(num_samples, latent_dim)
            library_sizes = torch.tensor(np.random.choice(size_factors, num_samples)).unsqueeze(1)
            batch_index = torch.zeros(num_samples, dtype=torch.int64)
            generated_data = model.module.generative(z=z_samples, batch_index=batch_index, library=library_sizes)
            cells = generated_data['px'].sample().cpu().numpy()

        cells_ag = cells
        cells = transform_gpu(cells, pca)

        logger.info(f"Shape of generated cells array: {cells.shape}")
        # Evaluate model
        avg_entropy, kl_div = evaluate_model(generated_data=cells)
        logger.info(f"Average Entropy: {avg_entropy}, KL Divergence: {kl_div}")
        avg_entropies.append(avg_entropy)
        kl_divs.append(kl_div)

        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]

        logger.info("PCAing ground truth data...")
        pca_sampled_expression_data = transform_gpu(sampled_expression_data, pca)
        logger.info("Done.")

        if args.umap_embed:
            pca_sampled_expression_data, cells = umap_embed(pca_sampled_expression_data, cells)

        if args.plot_umap:
            if i == 0:
                plot_umap(
                    pca_sampled_expression_data,
                    cells,
                    plot_name=f"scvi_umap.png"
                )

        mmd = mmd_rbf(cells[:,:args.num_pca_dims], pca_sampled_expression_data[:,:args.num_pca_dims], gamma=args.mmd_gamma)
        mmds.append(mmd)
        logger.info(f"MMD: {mmd}")
        wass = compute_wass(cells[:,:args.num_pca_dims], pca_sampled_expression_data[:,:args.num_pca_dims], reg=args.wass_reg)
        wasss.append(wass)
        logger.info(f"Wass: {wass}")


        # All genes
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag, sampled_expression_data)
        logger.info(f"SCVI R^2: {r2}")
        logger.info(f"SCVI Pearson correlation: {pearson_corr}")
        logger.info(f"SCVI Spearman correlation: {spearman_corr}")
        scvi_r2s.append(r2)
        scvi_pears.append(pearson_corr)
        scvi_spears.append(spearman_corr)

        # HVGS
        print(cells_ag[:, hvgs])
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
        logger.info(f"SCVI HVGS R^2: {r2}")
        logger.info(f"SCVI HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"SCVI HVGS Spearman correlation: {spearman_corr}")
        scvi_r2s_hvg.append(r2)
        scvi_pears_hvg.append(pearson_corr)
        scvi_spears_hvg.append(spearman_corr)

        # Rare HVGS
        r2, pearson_corr, spearman_corr = compute_statistics(add_gaussian_noise(cells_ag[:, rare_hvgs]), sampled_expression_data[:, rare_hvgs])
        logger.info(f"SCVI Rare HVGS R^2: {r2}")
        logger.info(f"SCVI Rare HVGS Pearson correlation: {pearson_corr}")
        logger.info(f"SCVI Rare HVGS Spearman correlation: {spearman_corr}")
        scvi_r2s_hvg_rare.append(r2)
        scvi_pears_hvg_rare.append(pearson_corr)
        scvi_spears_hvg_rare.append(spearman_corr)

    scvi_r2s = np.array(scvi_r2s)
    scvi_pears = np.array(scvi_pears)
    scvi_spears = np.array(scvi_spears)
    mmds = np.array(mmds)
    wasss = np.array(wasss)
    logger.info(f"MMD Mean {mmds.mean()} STD {mmds.std()}")
    logger.info(f"2-Wasserstein Mean {wasss.mean()} STD {wasss.std()}\n")

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

    avg_entropies = np.array(avg_entropies)
    kl_divs = np.array(kl_divs)
    logger.info(f"Average Entropy Mean {avg_entropies.mean()} STD {avg_entropies.std()}")
    logger.info(f"KL Divergence Mean {kl_divs.mean()} STD {kl_divs.std()}")

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
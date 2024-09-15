import argparse
import logging
import os
import pickle
import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
from utils.metrics import mmd_rbf, compute_wass, transform_gpu, umap_embed, evaluate_model, total_variation_distance
from scipy.sparse import issparse
from scvi.model import SCVI

from compute_mmd_wass_ifm import compute_kl_divergence_and_tv, assign_labels_to_generated_data

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_dir", type=str, default="/home/dfl32/scratch/training-runs/")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--mmd_gammas", type=float, nargs='+', default=[2.0])
    parser.add_argument("--wass_regs", type=float, nargs='+', default=[0.01])
    parser.add_argument("--input_dim", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_pca_dims", type=int, default=1000)
    parser.add_argument("--umap_embed", action="store_true")
    parser.add_argument("--num_umap_dims", type=int, default=10)
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of times to repeat the generation and sampling process")
    return parser.parse_args()

def main(args):
    # # Prep data
    # adata = sc.read_h5ad("/home/dfl32/project/ifm/cinemaot_data/raw_cinemaot.h5ad")
    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    # adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    # if issparse(adata.X):
    #     expression_data = adata.X.toarray()
    # else:
    #     expression_data = adata.X

    adata = sc.read_h5ad("/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/integrated_ifm_leiden.h5ad")
    expression_data = adata.X


    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"pcadim{args.input_dim}_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load SCVI model
    model = SCVI.load(
        dir_path="/home/dfl32/project/ifm/scvi_models/",
        prefix="epoch99_layers2_latent10_hidden128_calmflow",
        adata=adata
    )

    latent_dim = 10
    total_counts = adata.obs['total_counts']
    median_counts = total_counts.median()
    size_factors = total_counts / median_counts

    batch_size = args.batch_size
    num_steps = args.num_samples // batch_size

    cluster_tv_distances = []
    classifier_entropies = []
    classifier_kl_divs = []
    classifier_tv_distances = []
    label_kl_divs = []

    for repeat in range(args.num_repeats):
        logger.info(f"Starting repeat {repeat + 1}/{args.num_repeats}")
        
        with torch.no_grad():
            cells = []
            for step in tqdm(range(num_steps)):
                z_samples = torch.randn(batch_size, latent_dim).to(device)
                library_sizes = torch.tensor(np.random.choice(size_factors, batch_size)).unsqueeze(1).to(device)
                batch_index = torch.zeros(batch_size, dtype=torch.int64).to(device)
                generated_data = model.module.generative(z=z_samples, batch_index=batch_index, library=library_sizes)
                cells_batch = generated_data['px'].sample().cpu().numpy()
                cells.append(cells_batch)
            cells = np.concatenate(cells, axis=0)

        cells_ag = cells
        cells = transform_gpu(cells, pca)

        sample_indices = np.random.choice(expression_data.shape[0], size=args.num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]
        sampled_labels = adata.obs["leiden"].values[sample_indices].astype(int)
        
        logger.info("PCAing ground truth data...")
        pca_sampled_expression_data = transform_gpu(sampled_expression_data, pca)
        logger.info("Done.")

        generated_labels = assign_labels_to_generated_data(pca_sampled_expression_data, sampled_labels, cells)

        kl_div_leiden, cluster_tv_distance = compute_kl_divergence_and_tv(sampled_labels, generated_labels)
        cluster_tv_distances.append(cluster_tv_distance)
        logger.info(f"Cluster KL Divergence: {kl_div_leiden}")
        logger.info(f"Cluster Total Variation Distance: {cluster_tv_distance}")

        avg_entropy, kl_div, tv_distance, label_kl_div = evaluate_model(generated_data=cells)
        classifier_entropies.append(avg_entropy)
        classifier_kl_divs.append(kl_div)
        classifier_tv_distances.append(tv_distance)
        label_kl_divs.append(label_kl_div)

        logger.info(f"Classifier Average Entropy: {avg_entropy}")
        logger.info(f"Classifier KL Divergence: {kl_div}")
        logger.info(f"Classifier Total Variation Distance: {tv_distance}")
        logger.info(f"Label Distribution KL Divergence: {label_kl_div}")

    mean_cluster_tv = np.mean(cluster_tv_distances)
    std_cluster_tv = np.std(cluster_tv_distances)
    mean_classifier_entropy = np.mean(classifier_entropies)
    std_classifier_entropy = np.std(classifier_entropies)
    mean_classifier_kl_div = np.mean(classifier_kl_divs)
    std_classifier_kl_div = np.std(classifier_kl_divs)
    mean_classifier_tv = np.mean(classifier_tv_distances)
    std_classifier_tv = np.std(classifier_tv_distances)
    mean_label_kl_div = np.mean(label_kl_divs)
    std_label_kl_div = np.std(label_kl_divs)

    logger.info(f"Mean Cluster Total Variation Distance: {mean_cluster_tv} ± {std_cluster_tv}")
    logger.info(f"Mean Classifier Average Entropy: {mean_classifier_entropy} ± {std_classifier_entropy}")
    logger.info(f"Mean Classifier KL Divergence: {mean_classifier_kl_div} ± {std_classifier_kl_div}")
    logger.info(f"Mean Classifier Total Variation Distance: {mean_classifier_tv} ± {std_classifier_tv}")
    logger.info(f"Mean Label Distribution KL Divergence: {mean_label_kl_div} ± {std_label_kl_div}")

    # if args.umap_embed:
    #     pca_sampled_expression_data, cells = umap_embed(pca_sampled_expression_data, cells)

    # for gamma in args.mmd_gammas:
    #     mmd = mmd_rbf(cells[:, :args.num_umap_dims], pca_sampled_expression_data[:, :args.num_umap_dims], gamma=gamma)
    #     logger.info(f"MMD (gamma={gamma}): {mmd}")

    # for reg in args.wass_regs:
    #     wass = compute_wass(cells[:, :args.num_umap_dims], pca_sampled_expression_data[:, :args.num_umap_dims], reg=reg)
    #     logger.info(f"2-Wasserstein (reg={reg}): {wass}")

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
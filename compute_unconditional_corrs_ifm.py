import argparse
import json
import logging
import os
import pickle
from tqdm import tqdm
import pandas as pd

import numpy as np
import scanpy as sc
import safetensors
import torch
from torch import nn
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

from utils.modules import CustomVAEDecoder, TwoLayerMLP
from utils.metrics import mmd_rbf, compute_wass, transform_gpu, umap_embed, evaluate_model
from utils.plots import plot_umap

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cp_dir",
        type=str,
        default="/home/dfl32/scratch/training-runs/"
    )
    parser.add_argument(
        "--pretrained_weights",
        action="store_true",
    )
    parser.add_argument(
        "--model_json_path",
        type=str,
        default="/home/dfl32/project/ifm/models/ifm_paths.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pythia-160m-timepoints16-straightpathTrue-drop0.0ifm-2024-06-06_00-39-44"
    )
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
        "--temp",
        type=float,
        nargs='+',  # Allow multiple values
        default=[1.0]
    )
    parser.add_argument(
        "--hvgs",
        type=int,
        default=200
    )
    parser.add_argument(
        "--full_cell",
        action="store_true",
    )
    parser.add_argument(
        "--z_score",
        action="store_true",
    )
    parser.add_argument(
        "--time_points",
        type=int,
        default=16
    )
    parser.add_argument(
        "--space_dim",
        type=int,
        default=1
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1
    )
    parser.add_argument(
        "--reshape_postvae",
        action="store_true",
    )
    parser.add_argument(
        "--mlp_enc",
        action="store_true",
    )
    parser.add_argument(
        "--mlp_musig",
        action="store_true",
    )
    parser.add_argument(
        "--idfm",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100
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
        "--points_per_sample",
        type=int,
        default=1
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0
    )
    return parser.parse_args()

def multipoint_reshape(X, points_per_sample):

    batch_size, seq_len, feature_dim = X.shape

    new_batch_size = batch_size//points_per_sample

    # Reshape and permute the tensor
    x_reshaped = X.view(new_batch_size, points_per_sample, seq_len, feature_dim).permute(0, 2, 1, 3).reshape(new_batch_size, points_per_sample*seq_len, feature_dim)
    return x_reshaped

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

def z_score_norm(matrix):
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)

    stds[stds == 0] = 1

    # Perform z-score normalization
    normalized_matrix = (matrix - means) / stds
    return normalized_matrix


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

    # Set number of repeats and samples
    num_repeats = args.num_repeats
    num_samples = args.num_samples

    # Set device
    device = torch.device("cuda")

    # Set loop parameters
    batch_size = args.batch_size
    num_steps = num_samples//batch_size

    ### IFM ###
    # Load IFM model
    with open(args.model_json_path, "r") as f:
        model_paths = json.load(f)
    
    weights = "pretrained_weights" if args.pretrained_weights else "random_weights"
    if weights in model_paths:
        cp_path = model_paths[weights][str(args.space_dim)]
    else:
        beta_str = str(args.beta)
        cp_path = model_paths[beta_str][str(args.space_dim)]
    
    logger.info(f"CHECKPOINT PATH: {cp_path}")
    if args.space_dim == 640:
        args.space_dim = 1

    config_path = os.path.join(cp_path, "config.json")
    config = GPTNeoXConfig.from_pretrained(config_path)

    input_dim = args.input_dim
    model = GPTNeoXForCausalLM(config).to(device)
    if args.mlp_enc:
        model.cell_enc = TwoLayerMLP(input_dim, model.config.hidden_size*args.space_dim).to(device)
    else:
        model.cell_enc = nn.Linear(input_dim, model.config.hidden_size*args.space_dim).to(device)
    model.cell_dec = CustomVAEDecoder(
        hidden_size=config.hidden_size,
        input_dim=input_dim,
        device=device,
        reshape_postvae=args.reshape_postvae,
        space_dim=args.space_dim,
        num_blocks=1,
        mlp_enc=args.mlp_musig
    )


    model_weights_path = os.path.join(cp_path, "model.safetensors")
    pt_state_dict = safetensors.torch.load_file(model_weights_path, device="cuda")
    logger.info(model.load_state_dict(pt_state_dict))
    model.eval()

    time_points = args.time_points
    if args.idfm:
        euler_step_size = 1/(time_points-1)

    metrics = []

    for temp in args.temp:
        logger.info(f"Running inference with temperature: {temp}")
        ifm_r2s = []
        ifm_pears = []
        ifm_spears = []
        ifm_r2s_hvg = []
        ifm_pears_hvg = []
        ifm_spears_hvg = []
        ifm_r2s_hvg_rare = []
        ifm_pears_hvg_rare = []
        ifm_spears_hvg_rare = []
        mmds = []
        wasss = []
        avg_entropies = []
        kl_divs = []
        for i in range(num_repeats):
            with torch.no_grad():
                cells = []
                for step in tqdm(range(num_steps)):
                    inputs = torch.normal(0.0, 1.0, size=(batch_size, args.points_per_sample, input_dim)).to(device)
                    for _ in range(time_points-1):
                        outputs = model.cell_enc(inputs)

                        # Reshape for spatial integration
                        batch_size, seq_len, feature = outputs.shape
                        outputs = outputs.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
                        outputs = outputs.view(batch_size, seq_len* args.space_dim, feature // args.space_dim)


                        outputs = model.gpt_neox(inputs_embeds=outputs).last_hidden_state

                        if not args.reshape_postvae:
                            outputs = outputs.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
                            outputs = outputs.view(batch_size, seq_len, feature)

                        outputs, _, _ = model.cell_dec(outputs, temperature=temp)
                        last_outputs = outputs[:, -args.points_per_sample:, :]
                        if args.idfm:
                            last_outputs = inputs[:, -args.points_per_sample:, :] + (euler_step_size * last_outputs)
                        inputs = torch.concat([inputs, last_outputs], axis=1)
                    batch_size, _, feature_dim = outputs.shape
                    cells.append(outputs[:, -args.points_per_sample:, :].reshape(args.points_per_sample*batch_size, feature_dim).detach().cpu().numpy())
                cells = np.concatenate(cells, axis=0)

            # Evaluate model
            avg_entropy, kl_div = evaluate_model(generated_data=cells)
            logger.info(f"Average Entropy: {avg_entropy}, KL Divergence: {kl_div}")
            avg_entropies.append(avg_entropy)
            kl_divs.append(kl_div)

            logger.info("Inverse transforming IFM generated cells...")
            cells_ag = inverse_transform_gpu(cells, pca)
            # cells_ag = pca.inverse_transform(cells)
            logger.info("Done.")
            sample_indices = np.random.choice(expression_data.shape[0], size=num_samples*args.points_per_sample, replace=False)
            sampled_expression_data = expression_data[sample_indices]
            logger.info("PCAing ground truth data...")
            pca_sampled_expression_data = transform_gpu(sampled_expression_data, pca)
            logger.info("Done.")

            if args.plot_umap:
                if i == 0:
                    logger.info("Plotting UMAP...")
                    cp_path_parts = cp_path.split("/")
                    for part in cp_path_parts:
                        if "pythia-" in part:
                            checkpoint_base_name = part
                            break
                    if "checkpoint-" in cp_path_parts[-1]:
                        checkpoint_base_name += f"-{cp_path_parts[-1]}"
                    if not args.pretrained_weights:
                        checkpoint_base_name += f"-beta{args.beta}"
                    plot_umap(
                        pca_sampled_expression_data,
                        cells,
                        plot_name=f"{checkpoint_base_name}_temp{temp}_umap.png"
                    )

            if args.umap_embed:
                pca_sampled_expression_data, cells = umap_embed(pca_sampled_expression_data, cells)
            mmd = mmd_rbf(cells[:,:args.num_pca_dims], pca_sampled_expression_data[:,:args.num_pca_dims], gamma=args.mmd_gamma)
            mmds.append(mmd)
            logger.info(f"MMD: {mmd}")
            wass = compute_wass(cells[:,:args.num_pca_dims], pca_sampled_expression_data[:,:args.num_pca_dims], reg=args.wass_reg)
            wasss.append(wass)
            logger.info(f"Wass: {wass}")

            if args.z_score:
                logger.info("Normalizing genes by Z-score...")
                cells_ag = z_score_norm(cells_ag)
                sampled_expression_data = z_score_norm(sampled_expression_data)

            logger.info("Computing metrics...")
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

        logger.info(f"\nTemperature {temp}")
        ifm_r2s = np.array(ifm_r2s)
        ifm_pears = np.array(ifm_pears)
        ifm_spears = np.array(ifm_spears)
        mmds = np.array(mmds)
        wasss = np.array(wasss)
        avg_entropies = np.array(avg_entropies)
        kl_divs = np.array(kl_divs)
        ifm_r2s_hvg = np.array(ifm_r2s_hvg)  # Convert to NumPy array
        ifm_pears_hvg = np.array(ifm_pears_hvg)  # Convert to NumPy array
        ifm_spears_hvg = np.array(ifm_spears_hvg)  # Convert to NumPy array
        ifm_r2s_hvg_rare = np.array(ifm_r2s_hvg_rare)  # Convert to NumPy array
        ifm_pears_hvg_rare = np.array(ifm_pears_hvg_rare)  # Convert to NumPy array
        ifm_spears_hvg_rare = np.array(ifm_spears_hvg_rare)  # Convert to NumPy array

        # Handle edge case for num_repeats == 1
        if args.num_repeats > 1:
            mmd_mean, mmd_std = mmds.mean(), mmds.std()
            wass_mean, wass_std = wasss.mean(), wasss.std()
            avg_entropy_mean, avg_entropy_std = avg_entropies.mean(), avg_entropies.std()
            kl_div_mean, kl_div_std = kl_divs.mean(), kl_divs.std()
            ifm_r2_mean, ifm_r2_std = ifm_r2s.mean(), ifm_r2s.std()
            ifm_pears_mean, ifm_pears_std = ifm_pears.mean(), ifm_pears.std()
            ifm_spears_mean, ifm_spears_std = ifm_spears.mean(), ifm_spears.std()
            ifm_r2_hvg_mean, ifm_r2_hvg_std = ifm_r2s_hvg.mean(), ifm_r2s_hvg.std()
            ifm_pears_hvg_mean, ifm_pears_hvg_std = ifm_pears_hvg.mean(), ifm_pears_hvg.std()
            ifm_spears_hvg_mean, ifm_spears_hvg_std = ifm_spears_hvg.mean(), ifm_spears_hvg.std()
            ifm_r2_hvg_rare_mean, ifm_r2_hvg_rare_std = ifm_r2s_hvg_rare.mean(), ifm_r2s_hvg_rare.std()
            ifm_pears_hvg_rare_mean, ifm_pears_hvg_rare_std = ifm_pears_hvg_rare.mean(), ifm_pears_hvg_rare.std()
            ifm_spears_hvg_rare_mean, ifm_spears_hvg_rare_std = ifm_spears_hvg_rare.mean(), ifm_spears_hvg_rare.std()
        else:
            mmd_mean, mmd_std = mmds.mean(), 0
            wass_mean, wass_std = wasss.mean(), 0
            avg_entropy_mean, avg_entropy_std = avg_entropies.mean(), 0
            kl_div_mean, kl_div_std = kl_divs.mean(), 0
            ifm_r2_mean, ifm_r2_std = ifm_r2s.mean(), 0
            ifm_pears_mean, ifm_pears_std = ifm_pears.mean(), 0
            ifm_spears_mean, ifm_spears_std = ifm_spears.mean(), 0
            ifm_r2_hvg_mean, ifm_r2_hvg_std = ifm_r2s_hvg.mean(), 0
            ifm_pears_hvg_mean, ifm_pears_hvg_std = ifm_pears_hvg.mean(), 0
            ifm_spears_hvg_mean, ifm_spears_hvg_std = ifm_spears_hvg.mean(), 0
            ifm_r2_hvg_rare_mean, ifm_r2_hvg_rare_std = ifm_r2s_hvg_rare.mean(), 0
            ifm_pears_hvg_rare_mean, ifm_pears_hvg_rare_std = ifm_pears_hvg_rare.mean(), 0
            ifm_spears_hvg_rare_mean, ifm_spears_hvg_rare_std = ifm_spears_hvg_rare.mean(), 0

        logger.info(f"MMD Mean {mmd_mean} STD {mmd_std}")
        logger.info(f"2-Wasserstein Mean {wass_mean} STD {wass_std}\n")
        logger.info(f"Average Entropy Mean {avg_entropy_mean} STD {avg_entropy_std}")
        logger.info(f"KL Divergence Mean {kl_div_mean} STD {kl_div_std}")

        logger.info(f"IFM R^2 Mean {ifm_r2_mean} STD {ifm_r2_std}")
        logger.info(f"IFM Pearson Mean {ifm_pears_mean} STD {ifm_pears_std}")
        logger.info(f"IFM Spearman Mean {ifm_spears_mean} STD {ifm_spears_std}")

        logger.info(f"IFM HVGS R^2 Mean {ifm_r2_hvg_mean} STD {ifm_r2_hvg_std}")
        logger.info(f"IFM HVGS Pearson Mean {ifm_pears_hvg_mean} STD {ifm_pears_hvg_std}")
        logger.info(f"IFM HVGS Spearman Mean {ifm_spears_hvg_mean} STD {ifm_spears_hvg_std}")

        logger.info(f"IFM Rare HVGS R^2 Mean {ifm_r2_hvg_rare_mean} STD {ifm_r2_hvg_rare_std}")
        logger.info(f"IFM Rare HVGS Pearson Mean {ifm_pears_hvg_rare_mean} STD {ifm_pears_hvg_rare_std}")
        logger.info(f"IFM Rare HVGS Spearman Mean {ifm_spears_hvg_rare_mean} STD {ifm_spears_hvg_rare_std}")

        # Store metrics for the current temperature
        metrics.append({
            'Temperature': temp,
            'MMD Mean': mmd_mean, 'MMD STD': mmd_std,
            '2-Wasserstein Mean': wass_mean, '2-Wasserstein STD': wass_std,
            'Average Entropy Mean': avg_entropy_mean, 'Average Entropy STD': avg_entropy_std,
            'KL Divergence Mean': kl_div_mean, 'KL Divergence STD': kl_div_std,
            'IFM R^2 Mean': ifm_r2_mean, 'IFM R^2 STD': ifm_r2_std,
            'IFM Pearson Mean': ifm_pears_mean, 'IFM Pearson STD': ifm_pears_std,
            'IFM Spearman Mean': ifm_spears_mean, 'IFM Spearman STD': ifm_spears_std,
            'IFM HVGS R^2 Mean': ifm_r2_hvg_mean, 'IFM HVGS R^2 STD': ifm_r2_hvg_std,
            'IFM HVGS Pearson Mean': ifm_pears_hvg_mean, 'IFM HVGS Pearson STD': ifm_pears_hvg_std,
            'IFM HVGS Spearman Mean': ifm_spears_hvg_mean, 'IFM HVGS Spearman STD': ifm_spears_hvg_std,
            'IFM Rare HVGS R^2 Mean': ifm_r2_hvg_rare_mean, 'IFM Rare HVGS R^2 STD': ifm_r2_hvg_rare_std,
            'IFM Rare HVGS Pearson Mean': ifm_pears_hvg_rare_mean, 'IFM Rare HVGS Pearson STD': ifm_pears_hvg_rare_std,
            'IFM Rare HVGS Spearman Mean': ifm_spears_hvg_rare_mean, 'IFM Rare HVGS Spearman STD': ifm_spears_hvg_rare_std
        })

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'metrics_outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrame to CSV
    checkpoint_name = os.path.basename(args.checkpoint)
    output_path = os.path.join(output_dir, f"{checkpoint_name}_metrics.csv")
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Metrics saved to {output_path}")

    # Log top 5 ranked means for each metric
    metrics_to_rank = [
        'MMD Mean', '2-Wasserstein Mean', 'Average Entropy Mean', 'KL Divergence Mean', 'IFM R^2 Mean', 'IFM Pearson Mean', 'IFM Spearman Mean',
        'IFM HVGS R^2 Mean', 'IFM HVGS Pearson Mean', 'IFM HVGS Spearman Mean',
        'IFM Rare HVGS R^2 Mean', 'IFM Rare HVGS Pearson Mean', 'IFM Rare HVGS Spearman Mean'
    ]

    for metric in metrics_to_rank:
        logger.info(f"Top 5 temperatures for {metric}:")
        ranked_metrics = metrics_df[['Temperature', metric, metric.replace('Mean', 'STD')]].sort_values(by=metric, ascending=False).head(5)
        for _, row in ranked_metrics.iterrows():
            logger.info(f"Temperature: {row['Temperature']}, {metric}: {row[metric]}, STD: {row[metric.replace('Mean', 'STD')]}")

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
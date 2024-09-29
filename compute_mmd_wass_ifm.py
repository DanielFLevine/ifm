import argparse
import json
import logging
import os
import pickle
import numpy as np
import safetensors
import torch
from torch import nn
from tqdm import tqdm
from utils.metrics import mmd_rbf, compute_wass, transform_gpu, umap_embed, evaluate_model, total_variation_distance, mmd_rbf_adaptive
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from utils.modules import CustomVAEDecoder, TwoLayerMLP
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import umap
import matplotlib.pyplot as plt

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_dir", type=str, default="/home/dfl32/scratch/training-runs/")
    parser.add_argument("--model_json_path", type=str, default="/home/dfl32/project/ifm/models/ifm_paths.json")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument('--leiden', type=int, default=3)
    parser.add_argument("--mmd_gammas", type=float, nargs='+', default=[2.0])
    parser.add_argument("--wass_regs", type=float, nargs='+', default=[0.01])
    parser.add_argument("--space_dim", type=int, default=1)
    parser.add_argument("--input_dim", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--time_points", type=int, default=16)
    parser.add_argument("--points_per_sample", type=int, default=1)
    parser.add_argument("--pretrained_weights", action="store_true")
    parser.add_argument("--mlp_enc", action="store_true")
    parser.add_argument("--mlp_musig", action="store_true")
    parser.add_argument("--reshape_postvae", action="store_true")
    parser.add_argument("--umap_embed", action="store_true")
    parser.add_argument("--idfm", action="store_true")
    parser.add_argument("--z_score", action="store_true")
    parser.add_argument("--num_umap_dims", type=int, default=10)
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for decoding step")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of times to repeat the generation and sampling process")
    parser.add_argument("--attention_dropout", type=float, default=0.0, help="Attention dropout")
    return parser.parse_args()

def z_score_norm(matrix):
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)

    stds[stds == 0] = 1

    # Perform z-score normalization
    normalized_matrix = (matrix - means) / stds
    return normalized_matrix

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

def compute_kl_divergence_and_tv(sampled_labels, generated_labels):
    sampled_label_counts = np.bincount(sampled_labels)
    generated_label_counts = np.bincount(generated_labels, minlength=len(sampled_label_counts))

    sampled_distribution = sampled_label_counts / np.sum(sampled_label_counts)
    generated_distribution = generated_label_counts / np.sum(generated_label_counts)

    # Log the probabilities of each label
    logger.info(f"Sampled distribution: {dict(enumerate(sampled_distribution))}")
    logger.info(f"Generated distribution: {dict(enumerate(generated_distribution))}")

    kl_div = entropy(sampled_distribution, generated_distribution)
    
    # Calculate total variation distance
    tv_distance = 0.5 * np.sum(np.abs(sampled_distribution - generated_distribution))

    return kl_div, tv_distance

def assign_labels_to_generated_data(sampled_data, sampled_labels, generated_data, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(sampled_data)
    distances, indices = nbrs.kneighbors(generated_data)
    
    generated_labels = []
    for dist, idx in zip(distances, indices):
        neighbor_labels = sampled_labels[idx]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        
        # Calculate weights based on distances (inverse of distance)
        weights = 1 / (dist + 1e-8)  # Add a small value to avoid division by zero
        weighted_counts = np.zeros_like(unique_labels, dtype=float)
        
        for label, weight in zip(neighbor_labels, weights):
            weighted_counts[unique_labels == label] += weight
        
        # Assign the label with the highest weighted count
        assigned_label = unique_labels[np.argmax(weighted_counts)]
        generated_labels.append(assigned_label)
    
    return np.array(generated_labels, dtype=int)  # Ensure labels are integers

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
    leiden = f"0{args.leiden}" if args.leiden < 10 else f"{args.leiden}"
    adata = sc.read_h5ad(f"/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/integrated_ifm_leiden_{leiden}.h5ad")
    expression_data = adata.X

    # Load saved PCA model
    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"pcadim{args.input_dim}_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)

    # Set device
    device = torch.device("cuda")

    # Load IFM model
    with open(args.model_json_path, "r") as f:
        model_paths = json.load(f)
    
    weights = "pretrained_weights" if args.pretrained_weights else "random_weights"
    cp_path = model_paths[weights][str(args.attention_dropout)]
    # cp_path = "/home/dfl32/scratch/training-runs/traincustomTrue-vaeTrue-klw0.3-EleutherAI/pythia-160m-idfmFalse-hdim256idim1024nheads8nblocks2-space5-pps1-preweightsTrue-pca1000-timepoints16-straightpathTrue-drop0.0ifm-2024-08-26_11-32-37"
    logger.info(f"CHECKPOINT PATH: {cp_path}")

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
        euler_step_size = 1 / (time_points - 1)

    batch_size = args.batch_size
    num_steps = args.num_samples // (batch_size * args.points_per_sample)

    assert args.num_samples % (batch_size * args.points_per_sample) == 0, "batch_size * points_per_sample must divide num_samples"

    classifier_entropies = []
    classifier_kl_divs = []
    classifier_tv_distances = []
    label_kl_divs = []
    weighted_tv_distances = []
    incep_scores = []
    plausibility_scores = []  # Add this line
    mmds = {gamma: [] for gamma in args.mmd_gammas}
    adaptive_mmds = []
    wassersteins = {reg: [] for reg in args.wass_regs}

    for repeat in range(args.num_repeats):
        logger.info(f"Starting repeat {repeat + 1}/{args.num_repeats}")
        
        with torch.no_grad():
            cells = []
            for step in tqdm(range(num_steps)):
                inputs = torch.normal(0.0, 1.0, size=(batch_size, args.points_per_sample, args.input_dim)).to(device)
                for _ in range(time_points - 1):
                    outputs = model.cell_enc(inputs)

                    # Reshape for spatial integration
                    batch_size, seq_len, feature = outputs.shape
                    outputs = outputs.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
                    outputs = outputs.view(batch_size, seq_len * args.space_dim, feature // args.space_dim)

                    outputs = model.gpt_neox(inputs_embeds=outputs).last_hidden_state

                    if not args.reshape_postvae:
                        outputs = outputs.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
                        outputs = outputs.view(batch_size, seq_len, feature)

                    outputs, _, _ = model.cell_dec(outputs, temperature=args.temp)
                    last_outputs = outputs[:, -args.points_per_sample:, :]
                    if args.idfm:
                        last_outputs = inputs[:, -args.points_per_sample:, :] + (euler_step_size * last_outputs)
                    inputs = torch.concat([inputs, last_outputs], axis=1)
                batch_size, _, feature_dim = outputs.shape
                cells.append(outputs[:, -args.points_per_sample:, :].reshape(args.points_per_sample * batch_size, feature_dim).detach().cpu().numpy())
            cells = np.concatenate(cells, axis=0)

        if repeat == 0:
            np.save('/home/dfl32/project/ifm/figure_unconditional_data/ifm_cells.npy', cells)

        sample_indices = np.random.choice(expression_data.shape[0], size=args.num_samples, replace=False)
        sampled_expression_data = expression_data[sample_indices]
        sampled_labels = adata.obs["leiden"].values[sample_indices].astype(int)  # Convert leiden labels to integers

        logger.info("PCAing ground truth data...")
        pca_sampled_expression_data = transform_gpu(sampled_expression_data, pca)
        logger.info("Done.")

        # generated_labels = assign_labels_to_generated_data(pca_sampled_expression_data, sampled_labels, cells)

        # kl_div_leiden, cluster_tv_distance = compute_kl_divergence_and_tv(sampled_labels, generated_labels)
        # cluster_tv_distances.append(cluster_tv_distance)
        # logger.info(f"Cluster KL Divergence: {kl_div_leiden}")
        # logger.info(f"Cluster Total Variation Distance: {cluster_tv_distance}")

        avg_entropy, kl_div, tv_distance, label_kl_div, weighted_tv_distance, incep_score, plausibility_score = evaluate_model(
            generated_data=cells,
            checkpoint_path=f'/home/dfl32/scratch/unconditional_classifier_combined_labels/checkpoints_hidden_dim_256_{args.leiden}/checkpoint_step8000.pth',
            adata_path=f'/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/integrated_ifm_leiden_{leiden}.h5ad'
        )
        classifier_entropies.append(avg_entropy)
        classifier_kl_divs.append(kl_div)
        classifier_tv_distances.append(tv_distance)
        label_kl_divs.append(label_kl_div)
        weighted_tv_distances.append(weighted_tv_distance)
        incep_scores.append(incep_score)
        plausibility_scores.append(plausibility_score)  # Add this line

        logger.info(f"Classifier Average Entropy: {avg_entropy}")
        logger.info(f"Classifier KL Divergence: {kl_div}")
        logger.info(f"Classifier Total Variation Distance: {tv_distance}")
        logger.info(f"Label Distribution KL Divergence: {label_kl_div}")
        logger.info(f"Weighted Total Variation Distance: {weighted_tv_distance}")
        logger.info(f"Inception Score: {incep_score}")
        logger.info(f"Plausibility Score: {plausibility_score}")  # Add this line

        logger.info("Inverse transforming IFM generated cells...")
        cells_ag = inverse_transform_gpu(cells, pca)

        if args.z_score:
            logger.info("Normalizing genes by Z-score...")
            cells_ag = z_score_norm(cells_ag)
            sampled_expression_data = z_score_norm(sampled_expression_data)

        logger.info("Computing metrics...")

        # UMAP fitting and transformation
        umap_model = umap.UMAP(n_components=2)
        umap_sampled = umap_model.fit_transform(pca_sampled_expression_data)
        umap_generated = umap_model.transform(cells)
        if repeat == 0:
        # Plotting
            plt.figure(figsize=(5, 4))
            plt.scatter(umap_sampled[:, 0], umap_sampled[:, 1], c='blue', label='Sampled', alpha=0.5, s=0.2)
            plt.scatter(umap_generated[:, 0], umap_generated[:, 1], c='red', label='Generated', alpha=0.5, s=0.2)
            plt.legend(markerscale=5)
            plt.title('Sampled vs Generated Data')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'/home/dfl32/project/ifm/umaps/ifm_leiden_{args.leiden}_sampled_vs_generated.png', bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(5, 4))
            scatter = plt.scatter(umap_sampled[:, 0], umap_sampled[:, 1], c=sampled_labels, cmap='viridis', alpha=0.5, s=0.5)
            plt.colorbar(scatter, label='Leiden Label')
            plt.title('Sampled Data by Leiden Label')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'/home/dfl32/project/ifm/umaps/ifm_leiden_{args.leiden}_sampled.png', bbox_inches='tight')
            plt.close()

        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(umap_generated[:, 0], umap_generated[:, 1], c=generated_labels, cmap='viridis', alpha=0.5, s=0.5)
        # plt.colorbar(scatter, label='Leiden Label')
        # plt.title('Generated Data by Leiden Label')
        # plt.savefig(f'/home/dfl32/project/ifm/umaps/ifm_leiden_{args.leiden}_generated.png')
        # plt.close()

        if args.umap_embed:
            pca_sampled_expression_data, cells = umap_embed(pca_sampled_expression_data, cells)

        for gamma in args.mmd_gammas:
            mmd = mmd_rbf(cells[:, :args.num_umap_dims], pca_sampled_expression_data[:, :args.num_umap_dims], gamma=gamma)
            mmds[gamma].append(mmd)
            logger.info(f"MMD (gamma={gamma}): {mmd}")

        # Add adaptive MMD computation
        adaptive_mmd = mmd_rbf_adaptive(cells[:, :args.num_umap_dims], pca_sampled_expression_data[:, :args.num_umap_dims], k=20)
        adaptive_mmds.append(adaptive_mmd)
        logger.info(f"Adaptive MMD (k=20): {adaptive_mmd}")

        for reg in args.wass_regs:
            wass = compute_wass(cells[:, :args.num_umap_dims], pca_sampled_expression_data[:, :args.num_umap_dims], reg=reg)
            wassersteins[reg].append(wass)
            logger.info(f"2-Wasserstein (reg={reg}): {wass}")

    # Calculate means and standard deviations
    mean_classifier_entropy = np.mean(classifier_entropies)
    std_classifier_entropy = np.std(classifier_entropies)
    mean_classifier_kl_div = np.mean(classifier_kl_divs)
    std_classifier_kl_div = np.std(classifier_kl_divs)
    mean_classifier_tv = np.mean(classifier_tv_distances)
    std_classifier_tv = np.std(classifier_tv_distances)
    mean_label_kl_div = np.mean(label_kl_divs)
    std_label_kl_div = np.std(label_kl_divs)
    mean_weighted_tv = np.mean(weighted_tv_distances)
    std_weighted_tv = np.std(weighted_tv_distances)
    mean_incep_score = np.mean(incep_scores)
    std_incep_score = np.std(incep_scores)
    mean_plausibility_score = np.mean(plausibility_scores)
    std_plausibility_score = np.std(plausibility_scores)
    mean_adaptive_mmd = np.mean(adaptive_mmds)
    std_adaptive_mmd = np.std(adaptive_mmds)

    # Log all means and standard deviations
    logger.info(f"Mean Classifier Average Entropy: {mean_classifier_entropy} ± {std_classifier_entropy}")
    logger.info(f"Mean Classifier KL Divergence: {mean_classifier_kl_div} ± {std_classifier_kl_div}")
    logger.info(f"Mean Classifier Total Variation Distance: {mean_classifier_tv} ± {std_classifier_tv}")
    logger.info(f"Mean Label Distribution KL Divergence: {mean_label_kl_div} ± {std_label_kl_div}")
    logger.info(f"Mean Weighted Total Variation Distance: {mean_weighted_tv} ± {std_weighted_tv}")
    logger.info(f"Mean Inception Score: {mean_incep_score} ± {std_incep_score}")
    logger.info(f"Mean Plausibility Score: {mean_plausibility_score} ± {std_plausibility_score}")
    logger.info(f"Mean Adaptive MMD (k=20): {mean_adaptive_mmd} ± {std_adaptive_mmd}")

    for gamma in args.mmd_gammas:
        mean_mmd = np.mean(mmds[gamma])
        std_mmd = np.std(mmds[gamma])
        logger.info(f"Mean MMD (gamma={gamma}): {mean_mmd} ± {std_mmd}")

    for reg in args.wass_regs:
        mean_wass = np.mean(wassersteins[reg])
        std_wass = np.std(wassersteins[reg])
        logger.info(f"Mean 2-Wasserstein (reg={reg}): {mean_wass} ± {std_wass}")

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
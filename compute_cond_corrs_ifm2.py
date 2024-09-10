import argparse
import json
import logging
import os
import pickle
from tqdm import tqdm
from datetime import datetime

import numpy as np
import scanpy as sc
import safetensors
import torch
from torch import nn
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from datasets import load_from_disk
import numpy as np
import anndata as ad
from ipdb import set_trace
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd

from utils.modules import CustomVAEDecoder, TwoLayerMLP
from utils.metrics import mmd_rbf, compute_wass, transform_gpu, umap_embed, binned_KL, umap_transform
from utils.plots import plot_umap
from utils.adata_dataset import TestLabelDataset
from utils.data_utils import drop_rare_classes, get_control_adata

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cp_dir",
        type=str,
        default="/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/training_runs"
    )
    parser.add_argument(
        "--pretrained_weights",
        action="store_true",
    )
    parser.add_argument(
        "--model_json_path",
        type=str,
        default="/home/sh2748/ifm/models/ifm_big_models.json"
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
        default=1.0
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
        default=768
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
        "--prompt_path",
        type=str,
        default="/home/dfl32/project/ifm/prompts/cinemaot_prompts.json",
        help="Path to json file containing prompts"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-160m",
        help="Hugging Face model name for pretrained model"
    )
    parser.add_argument(
        "--kl_bins",
        type=int,
        default=50,
        help="Number of bins to use for KL divergence computation"
    )
    parser.add_argument(
        "--pert_split",
        type=str,
        default="ct_pert",
        help="Which train/test split to use for conditional generation. Valid values are 'ct_pert' or 'chron'"
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

def build_and_tokenize_prefixes(inputs, prompts, tokenizer, max_length=1024, device='cuda'):
    batch_size = len(inputs['perturbation'])
    prefixes = []
    for i in range(batch_size):
        cell_type = inputs['cell_type'][i]
        perturbation = inputs['perturbation'][i]
        chronicity = inputs['chronicity'][i]
        if perturbation == 'No stimulation':
            prefix = prompts['prefix']['control'][0].format(cell_type=cell_type, chronicity=chronicity)
        else:
            prefix = prompts['prefix']['perturbation'][0].format(cell_type=cell_type, perturbation=perturbation, chronicity=chronicity)
        prefixes.append(prefix)
    
    tokenized = tokenizer(
        prefixes, 
        truncation=True, 
        max_length=max_length, 
        padding='longest', 
        return_tensors='pt',
    )
    
    return {
        'prefix_input_ids': tokenized['input_ids'].to(device),
        'prefix_attention_mask': tokenized['attention_mask'].to(device)
    }

def log_stats(logger, repeat, class_label, r2_prefix, r2, pearson_corr, spearman_corr):
    logger.info(f"Repeat {i:<10} "
            f"Class {class_label:<20} "
            f"IFM {r2_prefix}: {r2:>20.6f} "
            f"Pearson correlation: {pearson_corr:>30.6f} "
            f"Spearman correlation: {spearman_corr:>30.6f}")




def main(args):
    # Prep data
    # The adata is already preprocessed
    adata = sc.read_h5ad(f"/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/{args.pert_split}_split/test_data_{args.pert_split}_split.h5ad") # shape (5902, 21710)
    train_adata = sc.read_h5ad(f"/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/{args.pert_split}_split/train_data_{args.pert_split}_split.h5ad")
    
    n_cells_before_dropping = adata.shape[0]
    logger.info("Dropping classes occurring less than 1%...")
    adata = drop_rare_classes(adata)
    n_cells_after_dropping = adata.shape[0]
    logger.info(f"Done dropping. Before dropping: {n_cells_before_dropping} cells, after dropping: {n_cells_after_dropping}")
    
    # HVG and rare HVG
    sc.pp.highly_variable_genes(adata, n_top_genes=args.hvgs)
    hvgs = adata.var['highly_variable']
    rare_hvgs = hvgs & (adata.var['n_cells'] < args.n_cell_thresh)
    
    if issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X
    
    # Load saved PCA model. 
    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"{args.pert_split}_pcadim1000_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    logger.info(f"Loading PCA model from {save_path}...")
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)
        logger.info("Done.")

    # Set number of repeats
    num_repeats = args.num_repeats
    device = torch.device("cuda")

    ### IFM ###
    # Load IFM model
    with open(args.model_json_path, "r") as f:
        model_paths = json.load(f)
    
    weights = "pretrained_weights" if args.pretrained_weights else "random_weights"
    cp_path = model_paths[args.pert_split][weights][str(args.space_dim)]
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

    # Create save directories
    experiment_directory = os.path.join("/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/inference_results", f"{args.pert_split}_split",f"pretrained-{args.pretrained_weights}_space{args.space_dim}", f"temperature-{args.temp}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = os.path.join(experiment_directory, f"run_{current_time}")
    os.makedirs(run_directory, exist_ok=True)

    # Load labels
    test_dataset = TestLabelDataset(adata)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    label_combinations = test_dataset.get_unique_label_combinations()

    ifm_r2s = {i: [] for i in label_combinations}
    ifm_pears = {i: [] for i in label_combinations}
    ifm_spears = {i: [] for i in label_combinations}
    ifm_r2s_hvg = {i: [] for i in label_combinations}
    ifm_pears_hvg = {i: [] for i in label_combinations}
    ifm_spears_hvg = {i: [] for i in label_combinations}
    ifm_r2s_hvg_rare = {i: [] for i in label_combinations}
    ifm_pears_hvg_rare = {i: [] for i in label_combinations}
    ifm_spears_hvg_rare = {i: [] for i in label_combinations}
    binned_kls = {i: [] for i in label_combinations}
    
    ifm_r2s_delta = {i: [] for i in label_combinations}
    ifm_pears_delta = {i: [] for i in label_combinations}
    ifm_spears_delta = {i: [] for i in label_combinations}
    ifm_r2s_hvg_delta = {i: [] for i in label_combinations}
    ifm_pears_hvg_delta = {i: [] for i in label_combinations}
    ifm_spears_hvg_delta = {i: [] for i in label_combinations}
    ifm_r2s_hvg_rare_delta = {i: [] for i in label_combinations}
    ifm_pears_hvg_rare_delta = {i: [] for i in label_combinations}
    ifm_spears_hvg_rare_delta = {i: [] for i in label_combinations}
    
    mmds = []
    wasss = []
    time_points = args.time_points

    # Load prompts and the tokenizer
    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generation
    if args.idfm:
        euler_step_size = 1/(time_points-1)
    for i in range(num_repeats):
        with torch.no_grad():
            cells = []
            umap_labels = {
                "perturbation": [],
                "cell_type": [],
                "chronicity": [],
                "combined_labels": []
            }
            # Generate cells from label combinations in the test dataset
            for batch in tqdm(test_dataloader):
                # Prepare labels
                umap_labels["perturbation"].extend(batch["perturbation"])
                umap_labels["cell_type"].extend(batch["cell_type"])
                umap_labels["chronicity"].extend(batch["chronicity"])
                umap_labels["combined_labels"].extend(batch["combined_labels"])
                
                prefix_tokens = build_and_tokenize_prefixes(batch, prompts, tokenizer)
                prefix_input_ids = prefix_tokens['prefix_input_ids']
                prefix_tokens_attention_mask = prefix_tokens['prefix_attention_mask']
                prefix_length = prefix_input_ids.shape[1]
                batch_size = prefix_input_ids.shape[0]
                paths = torch.normal(0.0, 1.0, size=(batch_size, args.points_per_sample, input_dim)).to(device)
                prefix_emb = model.gpt_neox.embed_in(prefix_input_ids)
                
                # Autoregressive generation
                for _ in range(time_points-1):
                    path_emb = model.cell_enc(paths)
                    # Reshape for spatial integration
                    batch_size, seq_len, feature = path_emb.shape
                    path_emb = path_emb.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
                    path_emb = path_emb.view(batch_size, seq_len* args.space_dim, feature // args.space_dim)
                    
                    input_emb = torch.concat([prefix_emb, path_emb], dim=1)
                    attention_mask = torch.cat(
                            [
                                prefix_tokens_attention_mask,
                                torch.ones((path_emb.shape[0], path_emb.shape[1]), dtype=torch.int32).to(model.device)
                            ], 
                            dim=1
                    )
                    
                    outputs = model.gpt_neox(inputs_embeds=input_emb, attention_mask=attention_mask).last_hidden_state
                    outputs = outputs[:, prefix_length:, ...]
                    
                    if not args.reshape_postvae:
                        outputs = outputs.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
                        outputs = outputs.view(batch_size, seq_len, feature)
                    
                    outputs, _, _ = model.cell_dec(outputs, temperature=args.temp)
                    last_outputs = outputs[:, -args.points_per_sample:, :]
                    
                    if args.idfm:
                        last_outputs = paths[:, -args.points_per_sample:, :] + (euler_step_size * last_outputs)
                    
                    paths = torch.concat([paths, last_outputs], axis=1)
                    
                batch_size, _, feature_dim = outputs.shape
                cells.append(outputs[:, -args.points_per_sample:, :].reshape(args.points_per_sample*batch_size, feature_dim).detach().cpu().numpy())
            
            cells = np.concatenate(cells, axis=0)
        
        logger.info("Save generated cells...")
        np.save(os.path.join(run_directory, "generated_cells.npy"), cells) 
        logger.info("Done")

        logger.info("PCAing ground truth data...")
        pca_expression_data = transform_gpu(expression_data, pca)
        logger.info("Done.")

        logger.info("Inverse transforming IFM generated cells...")
        cells_ag = inverse_transform_gpu(cells, pca)
        logger.info("Done.")
        
        if args.plot_umap:
            umap_directory = os.path.join(run_directory, "UMAPs")
            os.makedirs(umap_directory, exist_ok=True)
            if i == 0:
                logger.info("Plotting UMAP...")
                plot_umap(
                    pca_expression_data,
                    cells,
                    save_dir=umap_directory,
                    plot_name=f"calmflow_pp4_space{args.space_dim}",
                    labels=umap_labels
                )

        
        # Compute MMD and Wass on PCA GT data and PCA gen data
        mmd = mmd_rbf(cells[:,:args.num_pca_dims], pca_expression_data[:,:args.num_pca_dims])
        mmds.append(mmd)
        logger.info(f"MMD: {mmd}")
        wass = compute_wass(cells[:,:args.num_pca_dims], pca_expression_data[:,:args.num_pca_dims])
        wasss.append(wass)
        logger.info(f"Wass: {wass}")
        
        if args.z_score:
            logger.info("Normalizing genes by Z-score...")
            cells_ag = z_score_norm(cells_ag)
            expression_data = z_score_norm(expression_data)

        
        logger.info("Computing metrics...")
        # Prepare for KL
        # Make directory, load UMAP model, define bins
        umap_model_dir = "/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/umap_models"
        umap_model_path = os.path.join(umap_model_dir, f"{args.pert_split}_split_umap_model.pkl")
        with open(umap_model_path, 'rb') as f:
            logger.info(f"Loading UMAP model from {umap_model_path}...")
            umap_model = pickle.load(f)
            logger.info("Done.")
        logger.info("Transforming PCA expression data and PCA IFM data to 2D UMAP...")
        umap_pca_expression_data = umap_transform(pca_expression_data, umap_model)
        umap_cells = umap_transform(cells, umap_model)
        logger.info("Done.")
        set_trace()
        assert umap_pca_expression_data.shape[1] == 2, f'Want 2D UMAP but got {umap_pca_expression_data.shape[1]}'
        bin_edges = [
            np.linspace(np.min(umap_pca_expression_data[:, i]), np.max(umap_pca_expression_data[:, i]), args.kl_bins + 1)
            for i in range(umap_pca_expression_data.shape[1])
        ]
        logger.info(f"Created bins whose dimension 0 from {bin_edges[0][0]} to {bin_edges[0][-1]} and dimension 1 from {bin_edges[1][0]} to {bin_edges[1][-1]}..")
        
        # Compute metrics per class
        for class_label in label_combinations:
            # Get data for this class
            class_indices = test_dataset.get_indices_of_class(class_label)
            class_cells_ag = cells_ag[class_indices]
            class_expression_data = expression_data[class_indices] # or adata[adata.obs["combined_labels"]==class_label].X, equivalent
            class_umap_pca_expression_data = umap_pca_expression_data[class_indices]
            class_umap_cells = umap_cells[class_indices]

            # Find the control cells of the same cell type and chronicity
            # Subtract the mean control cell from the cells to get delta
            control_adata = get_control_adata(train_adata, class_label)
            control_mean = np.mean(control_adata.X, axis=0) # shape (21710,)
            class_delta_cells_ag = class_cells_ag - control_mean
            class_delta_expression_data = class_expression_data - control_mean

            logger.info(f"Class {class_label} has {len(class_indices)} data points.")
            
            # IFM
            r2, pearson_corr, spearman_corr = compute_statistics(class_cells_ag, class_expression_data)
            delta_r2, delta_pearson_corr, delta_spearman_corr = compute_statistics(class_delta_cells_ag, class_delta_expression_data)
            kl_divergence = binned_KL(bin_edges = bin_edges, gt_data=class_umap_pca_expression_data, gen_data=class_umap_cells, class_label=class_label)
            logger.info(f"Repeat {str(i).ljust(5)} "
                f"Class {str(class_label).ljust(15)} "
                f"IFM R^2: {str(f'{r2:.6f}').ljust(10)} "
                f"Pearson correlation: {str(f'{pearson_corr:.6f}').ljust(10)} "
                f"Spearman correlation: {str(f'{spearman_corr:.6f}').ljust(10)}"
                f"Delta IFM R^2: {str(f'{delta_r2:.6f}').ljust(10)} "
                f"Delta Pearson correlation: {str(f'{delta_pearson_corr:.6f}').ljust(10)} "
                f"Delta Spearman correlation: {str(f'{delta_spearman_corr:.6f}').ljust(10)}"
                f"Binned 2D UMAP KL Divergence: {str(f'{kl_divergence:.6f}').ljust(10)} "
                )
            ifm_r2s[class_label].append(r2)
            ifm_pears[class_label].append(pearson_corr)
            ifm_spears[class_label].append(spearman_corr)
            ifm_r2s_delta[class_label].append(delta_r2)
            ifm_pears_delta[class_label].append(delta_pearson_corr)
            ifm_spears_delta[class_label].append(delta_spearman_corr)
            binned_kls[class_label].append(kl_divergence)

            # HVGS
            r2, pearson_corr, spearman_corr = compute_statistics(class_cells_ag[:, hvgs], class_expression_data[:, hvgs])
            delta_r2, delta_pearson_corr, delta_spearman_corr = compute_statistics(class_delta_cells_ag[:, hvgs], class_delta_expression_data[:, hvgs])
            logger.info(f"Repeat {str(i).ljust(5)} "
                f"Class {str(class_label).ljust(15)} "
                f"IFM HVGS R^2: {str(f'{r2:.6f}').ljust(10)} "
                f"Pearson correlation: {str(f'{pearson_corr:.6f}').ljust(10)} "
                f"Spearman correlation: {str(f'{spearman_corr:.6f}').ljust(10)} "
                f"Delta IFM HVGS R^2: {str(f'{delta_r2:.6f}').ljust(10)} "
                f"Delta Pearson correlation: {str(f'{delta_pearson_corr:.6f}').ljust(10)} "
                f"Delta Spearman correlation: {str(f'{delta_spearman_corr:.6f}').ljust(10)} ")
            ifm_r2s_hvg[class_label].append(r2)
            ifm_pears_hvg[class_label].append(pearson_corr)
            ifm_spears_hvg[class_label].append(spearman_corr)
            ifm_r2s_hvg_delta[class_label].append(delta_r2)
            ifm_pears_hvg_delta[class_label].append(delta_pearson_corr)
            ifm_spears_hvg_delta[class_label].append(delta_spearman_corr)

            # Rare HVGS
            r2, pearson_corr, spearman_corr = compute_statistics(class_cells_ag[:, rare_hvgs], class_expression_data[:, rare_hvgs])
            delta_r2, delta_pearson_corr, delta_spearman_corr = compute_statistics(class_delta_cells_ag[:, rare_hvgs], class_delta_expression_data[:, rare_hvgs])
            logger.info(f"Repeat {str(i).ljust(5)} "
                f"Class {str(class_label).ljust(15)} "
                f"IFM Rare HVGS R^2: {str(f'{r2:.6f}').ljust(10)} "
                f"Pearson correlation: {str(f'{pearson_corr:.6f}').ljust(10)} "
                f"Spearman correlation: {str(f'{spearman_corr:.6f}').ljust(10)} "
                f"Delta IFM Rare HVGS R^2: {str(f'{delta_r2:.6f}').ljust(10)} "
                f"Delta Pearson correlation: {str(f'{delta_pearson_corr:.6f}').ljust(10)} "
                f"Delta Spearman correlation: {str(f'{delta_spearman_corr:.6f}').ljust(10)} ")
            ifm_r2s_hvg_rare[class_label].append(r2)
            ifm_pears_hvg_rare[class_label].append(pearson_corr)
            ifm_spears_hvg_rare[class_label].append(spearman_corr)
            ifm_r2s_hvg_rare_delta[class_label].append(delta_r2)
            ifm_pears_hvg_rare_delta[class_label].append(delta_pearson_corr)
            ifm_spears_hvg_rare_delta[class_label].append(delta_spearman_corr)

    logger.info(f"\nTemperature {args.temp}")
    
    binned_kls = {class_label: np.array(binned_kls[class_label]) for class_label in binned_kls}
    mmds = np.array(mmds)
    wasss = np.array(wasss)
    logger.info(f"MMD Mean {mmds.mean():.6f} STD {mmds.std():.6f}")
    logger.info(f"2-Wasserstein Mean {wasss.mean():.6f} STD {wasss.std():.6f}\n")
    
    
    ifm_r2s = {class_label: np.array(ifm_r2s[class_label]) for class_label in ifm_r2s}
    ifm_pears = {class_label: np.array(ifm_pears[class_label]) for class_label in ifm_pears}
    ifm_spears = {class_label: np.array(ifm_spears[class_label]) for class_label in ifm_spears}
    
    ifm_r2s_hvg = {class_label: np.array(ifm_r2s_hvg[class_label]) for class_label in ifm_r2s_hvg}
    ifm_pears_hvg = {class_label: np.array(ifm_pears_hvg[class_label]) for class_label in ifm_pears_hvg}
    ifm_spears_hvg = {class_label: np.array(ifm_spears_hvg[class_label]) for class_label in ifm_spears_hvg}

    ifm_r2s_hvg_rare = {class_label: np.array(ifm_r2s_hvg_rare[class_label]) for class_label in ifm_r2s_hvg_rare}
    ifm_pears_hvg_rare = {class_label: np.array(ifm_pears_hvg_rare[class_label]) for class_label in ifm_pears_hvg_rare}
    ifm_spears_hvg_rare = {class_label: np.array(ifm_spears_hvg_rare[class_label]) for class_label in ifm_spears_hvg_rare}

    ifm_r2s_delta = {class_label: np.array(ifm_r2s_delta[class_label]) for class_label in ifm_r2s_delta}
    ifm_pears_delta = {class_label: np.array(ifm_pears_delta[class_label]) for class_label in ifm_pears_delta}
    ifm_spears_delta = {class_label: np.array(ifm_spears_delta[class_label]) for class_label in ifm_spears_delta}

    ifm_r2s_hvg_delta = {class_label: np.array(ifm_r2s_hvg_delta[class_label]) for class_label in ifm_r2s_hvg_delta}
    ifm_pears_hvg_delta = {class_label: np.array(ifm_pears_hvg_delta[class_label]) for class_label in ifm_pears_hvg_delta}
    ifm_spears_hvg_delta = {class_label: np.array(ifm_spears_hvg_delta[class_label]) for class_label in ifm_spears_hvg_delta}

    ifm_r2s_hvg_rare_delta = {class_label: np.array(ifm_r2s_hvg_rare_delta[class_label]) for class_label in ifm_r2s_hvg_rare_delta}
    ifm_pears_hvg_rare_delta = {class_label: np.array(ifm_pears_hvg_rare_delta[class_label]) for class_label in ifm_pears_hvg_rare_delta}
    ifm_spears_hvg_rare_delta = {class_label: np.array(ifm_spears_hvg_rare_delta[class_label]) for class_label in ifm_spears_hvg_rare_delta}
    # set_trace()
    for class_label in label_combinations:
        logger.info(f"Class {class_label} IFM R^2 {ifm_r2s[class_label].mean():.6f} +/- {ifm_r2s[class_label].std():.6f}, Pearson {ifm_pears[class_label].mean():.6f} +/- {ifm_pears[class_label].std():.6f}, Spearman {ifm_spears[class_label].mean():.6f} +/- {ifm_spears[class_label].std():.6f}, KL {binned_kls[class_label].mean():.6f} +/- {binned_kls[class_label].std():.6f}")
        logger.info(f"Class {class_label} IFM HVGS R^2 {ifm_r2s_hvg[class_label].mean():.6f} +/- {ifm_r2s_hvg[class_label].std():.6f}, Pearson {ifm_pears_hvg[class_label].mean():.6f} +/- {ifm_pears_hvg[class_label].std():.6f}, Spearman {ifm_spears_hvg[class_label].mean():.6f} +/- {ifm_spears_hvg[class_label].std():.6f}")
        logger.info(f"Class {class_label} IFM Rare HVGS R^2 {ifm_r2s_hvg_rare[class_label].mean():.6f} +/- {ifm_r2s_hvg_rare[class_label].std():.6f} Pearson {ifm_pears_hvg_rare[class_label].mean():.6f} +/- {ifm_pears_hvg_rare[class_label].std():.6f} Spearman {ifm_spears_hvg_rare[class_label].mean():.6f} +/- {ifm_spears_hvg_rare[class_label].std():.6f}")
        logger.info(f"Class {class_label} Delta IFM R^2 {ifm_r2s_delta[class_label].mean():.6f} +/- {ifm_r2s_delta[class_label].std():.6f}, Delta Pearson {ifm_pears_delta[class_label].mean():.6f} +/- {ifm_pears_delta[class_label].std():.6f}, Delta Spearman {ifm_spears_delta[class_label].mean():.6f} +/- {ifm_spears_delta[class_label].std():.6f}")
        logger.info(f"Class {class_label} Delta IFM HVGS R^2 {ifm_r2s_hvg_delta[class_label].mean():.6f} +/- {ifm_r2s_hvg_delta[class_label].std():.6f}, Delta Pearson {ifm_pears_hvg_delta[class_label].mean():.6f} +/- {ifm_pears_hvg_delta[class_label].std():.6f}, Delta Spearman {ifm_spears_hvg_delta[class_label].mean():.6f} +/- {ifm_spears_hvg_delta[class_label].std():.6f}")
        logger.info(f"Class {class_label} Delta IFM Rare HVGS R^2 {ifm_r2s_hvg_rare_delta[class_label].mean():.6f} +/- {ifm_r2s_hvg_rare_delta[class_label].std():.6f} Delta Pearson {ifm_pears_hvg_rare_delta[class_label].mean():.6f} +/- {ifm_pears_hvg_rare_delta[class_label].std():.6f} Delta Spearman {ifm_spears_hvg_rare_delta[class_label].mean():.6f} +/- {ifm_spears_hvg_rare_delta[class_label].std():.6f}")
    
    logger.info("Preparing metrics to a pandas dataframe...")
    # Prepare the data for the DataFrame
    data = []

    # Add MMD and Wasserstein metrics
    data.append(["Overall", "MMD Mean", mmds.mean(), mmds.std()])
    data.append(["Overall", "2-Wasserstein Mean", wasss.mean(), wasss.std()])

    # Add IFM metrics for each class label
    for class_label in label_combinations:
        data.append([class_label, "IFM R^2", ifm_r2s[class_label]])
        data.append([class_label, "IFM Pearson", ifm_pears[class_label]])
        data.append([class_label, "IFM Spearman", ifm_spears[class_label]])
        data.append([class_label, "IFM HVGS R^2", ifm_r2s_hvg[class_label]])
        data.append([class_label, "IFM HVGS Pearson", ifm_pears_hvg[class_label]])
        data.append([class_label, "IFM HVGS Spearman", ifm_spears_hvg[class_label]])
        data.append([class_label, "IFM Rare HVGS R^2", ifm_r2s_hvg_rare[class_label]])
        data.append([class_label, "IFM Rare HVGS Pearson", ifm_pears_hvg_rare[class_label]])
        data.append([class_label, "IFM Rare HVGS Spearman", ifm_spears_hvg_rare[class_label]])
        data.append([class_label, "Delta IFM R^2", ifm_r2s_delta[class_label]])
        data.append([class_label, "Delta IFM Pearson", ifm_pears_delta[class_label]])
        data.append([class_label, "Delta IFM Spearman", ifm_spears_delta[class_label]])
        data.append([class_label, "Delta IFM HVGS R^2", ifm_r2s_hvg_delta[class_label]])
        data.append([class_label, "Delta IFM HVGS Pearson", ifm_pears_hvg_delta[class_label]])
        data.append([class_label, "Delta IFM HVGS Spearman", ifm_spears_hvg_delta[class_label]])
        data.append([class_label, "Delta IFM Rare HVGS R^2", ifm_r2s_hvg_rare_delta[class_label]])
        data.append([class_label, "Delta IFM Rare HVGS Pearson", ifm_pears_hvg_rare_delta[class_label]])
        data.append([class_label, "Delta IFM Rare HVGS Spearman", ifm_spears_hvg_rare_delta[class_label]])
    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Class Label", "Metric", "Mean", "STD"])
    logger.info("Done.")

    
    # Define the CSV file path
    csv_file_path = os.path.join(run_directory, "computed_metrics.csv")
    logger.info(f"Saving computed metrics to {csv_file_path}")
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    logger.info("Done.")

    # Save the logger information to a file
    logger.info("Saving logger...")
    log_file_path = os.path.join(run_directory, "generation_log.txt")
    with open(log_file_path, "w") as log_file:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream = log_file
                break
        else:
            file_handler = logging.FileHandler(log_file_path)
            logger.addHandler(file_handler)


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
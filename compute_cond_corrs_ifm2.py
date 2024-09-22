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
import matplotlib.pyplot as plt
import umap

from utils.modules import CustomVAEDecoder, TwoLayerMLP
from utils.metrics import mmd_rbf, compute_wass, transform_gpu, umap_embed, binned_KL, inception_score, leiden_KL, mmd_rbf_adaptive
from utils.plots import plot_umap, plot_pca
from utils.adata_dataset import TestLabelDataset
from utils.data_utils import drop_rare_classes, get_control_adata
from utils.leiden_classifier import MLPClassifier

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
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="Flag to indicate whether to compute metrics"
    )
    parser.add_argument(
        "--leiden_resol",
        type=float,
        default=1.0,
        help="Resolution parameter for Leiden clustering"
    )
    parser.add_argument(
        "--leiden_classifier_path",
        type=str,
        default="/home/sh2748/ifm/models/leiden_classifier.json",
        help="Path to the JSON file containing paths to the trained Leiden classifier models"
    )
    parser.add_argument(
        "--adaptive_mmd_kernel_k",
        type=int,
        default=20,
        help="Kernel parameter k for adaptive MMD"
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

def load_pca_model(pert_split):
    save_dir = "/home/dfl32/project/ifm/projections"
    save_name = f"{pert_split}_pcadim1000_numsamples10000.pickle"
    save_path = os.path.join(save_dir, save_name)
    logger.info(f"Loading PCA model from {save_path}...")
    with open(save_path, 'rb') as f:
        pca = pickle.load(f)
        logger.info("Done.")
    return pca

def load_ifm_model(args, device):
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

    return model

def generate_ifm_pca_data(model, test_dataloader, prompts, tokenizer, args, device, time_points):
    ifm_pca_data = []
    umap_labels = {
        "perturbation": [],
        "cell_type": [],
        "chronicity": [],
        "combined_labels": []
    }
    # Generate ifm_pca_data from label combinations in the test dataset
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
        paths = torch.normal(0.0, 1.0, size=(batch_size, args.points_per_sample, args.input_dim)).to(device)
        prefix_emb = model.gpt_neox.embed_in(prefix_input_ids)
        
        # Autoregressive generation
        for _ in range(time_points-1):
            path_emb = model.cell_enc(paths)
            # Reshape for spatial integration
            batch_size, seq_len, feature = path_emb.shape
            # set_trace()
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
                raise NotImplementedError("IDFM not implemented")
                # last_outputs = paths[:, -args.points_per_sample:, :] + (euler_step_size * last_outputs)
            
            paths = torch.concat([paths, last_outputs], axis=1)
            
        batch_size, _, feature_dim = outputs.shape
        ifm_pca_data.append(outputs[:, -args.points_per_sample:, :].reshape(args.points_per_sample*batch_size, feature_dim).detach().cpu().numpy())
    # set_trace()
    ifm_pca_data = np.concatenate(ifm_pca_data, axis=0) # shape (n_cells, 1000)
    return ifm_pca_data, umap_labels # TODO: only return full_gt_adata

def load_umap_model(umap_pickle_name):
    # umap_pickle_name is something like "{pert_split}_split_umap_model.pkl"
    umap_model_dir = "/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/umap_models"
    umap_model_path = os.path.join(umap_model_dir, umap_pickle_name)
    with open(umap_model_path, 'rb') as f:
        umap_model = pickle.load(f)
    return umap_model

def main(args):
    # Create save directories
    experiment_directory = os.path.join("/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/inference_results", f"{args.pert_split}_split",f"pretrained-{args.pretrained_weights}_space{args.space_dim}", f"temperature-{args.temp}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_directory = os.path.join(experiment_directory, f"run_{current_time}")
    os.makedirs(run_directory, exist_ok=True)
    umap_directory = os.path.join(run_directory, "UMAPs")
    os.makedirs(umap_directory, exist_ok=True)
    logger.info(f"Saved to: {run_directory}")
    logger.info(f"Copy the above directory to inference_report.ipynb to see inference results!")

    # Prep data
    # The full_gt_adata is already preprocessed
    # Load GT test data. We are going to use its labels to generate and use its X to compute metrics
    # BUG: this is not PCA gt but full gt
    full_gt_adata = sc.read_h5ad(f"/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/{args.pert_split}_split/test_data_{args.pert_split}_split.h5ad") # shape (5902, 21710)
    # Load training data. We are going to find control ifm_pca_data from it.
    train_adata = sc.read_h5ad(f"/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/{args.pert_split}_split/train_data_{args.pert_split}_split.h5ad")

    full_gt_adata = drop_rare_classes(full_gt_adata, logger)
    
    if issparse(full_gt_adata.X):
        expression_data = full_gt_adata.X.toarray()
    else:
        expression_data = full_gt_adata.X
    
    # Load saved PCA model. 
    pca = load_pca_model(args.pert_split)

    num_repeats = args.num_repeats
    device = torch.device("cuda")

    ### IFM ###
    # Load IFM model
    model = load_ifm_model(args, device)

    # Create dataset and dataloader of labels.
    test_dataset = TestLabelDataset(full_gt_adata)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    label_combinations = test_dataset.get_unique_label_combinations()

    # Initialize arrays to store per-class metrics
    ifm_r2s = {i: [] for i in label_combinations}
    ifm_pears = {i: [] for i in label_combinations}
    ifm_spears = {i: [] for i in label_combinations}

    ifm_r2s_de = {i: [] for i in label_combinations}
    ifm_pears_de = {i: [] for i in label_combinations}
    ifm_spears_de = {i: [] for i in label_combinations}

    binned_kls = {i: [] for i in label_combinations}
    
    ifm_r2s_delta = {i: [] for i in label_combinations}
    ifm_pears_delta = {i: [] for i in label_combinations}
    ifm_spears_delta = {i: [] for i in label_combinations}

    ifm_r2s_de_delta = {i: [] for i in label_combinations}
    ifm_pears_de_delta = {i: [] for i in label_combinations}
    ifm_spears_de_delta = {i: [] for i in label_combinations}

    
    # Overall metrics
    umap10D_mmds = []
    umap10D_adaptive_mmds = []
    umap10D_wasss = []
    
    umap_combined10D_mmds = []
    umap_combined10D_adaptive_mmds = []
    umap_combined10D_wasss = []
    
    full_pca_mmds = []
    full_pca_adaptive_mmds = []
    full_pca_wasss = []
    
    first_10_pc_mmds = []
    first_10_pc_adaptive_mmds = []
    first_10_pc_wasss = []

    leiden_kls = []
    incep_scores = []
    overall_binned_kls = []
    
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
            ifm_pca_data, umap_labels = generate_ifm_pca_data(model, test_dataloader, prompts, tokenizer, args, device, time_points)
        
        #TODO: will put this all into generate_ifm_pca_data()
        logger.info("Save generated ifm_pca_data as anndata...")
        ifm_pca_adata = ad.AnnData(X=ifm_pca_data) # Already PCA'd
        ifm_pca_adata.obs["perturbation"] = umap_labels["perturbation"]
        ifm_pca_adata.obs["cell_type"] = umap_labels["cell_type"]
        ifm_pca_adata.obs["chronicity"] = umap_labels["chronicity"]
        ifm_pca_adata.obs["combined_labels"] = umap_labels["combined_labels"]
        ifm_pca_adata_save_path = os.path.join(run_directory, "generated_cells_pca.h5ad")
        ifm_pca_adata.write(ifm_pca_adata_save_path)
        logger.info(f"Saved to {ifm_pca_adata_save_path}")
        
        logger.info("Plotting UMAP for generated data...")
        sc.pp.neighbors(ifm_pca_adata, use_rep='X')
        sc.tl.umap(ifm_pca_adata)
        sc.settings.figdir = umap_directory
        sc.pl.umap(ifm_pca_adata, color='combined_labels', save='_generated_cells_umap_alone.png', 
                   show=False, title='UMAP of Generated Data', size=50)
        # Remove UMAP coordinates from the AnnData object
        del ifm_pca_adata.obsm['X_umap']
        del ifm_pca_adata.uns['neighbors']
        logger.info("Done")

        logger.info("PCAing ground truth data...")
        gt_pca_data = transform_gpu(expression_data, pca) # num_samples x 1000, np array
        gt_pca_adata = ad.AnnData(X=gt_pca_data)
        gt_pca_adata.obs = full_gt_adata.obs.copy()
        logger.info("Plotting UMAP for generated data...")
        sc.pp.neighbors(gt_pca_adata, use_rep='X')
        sc.tl.umap(gt_pca_adata)
        sc.settings.figdir = umap_directory
        sc.pl.umap(gt_pca_adata, color='combined_labels', save='_gt_cells_umap_alone.png', 
                   show=False, title='UMAP of Ground Truth Data')

        # Remove UMAP coordinates from the AnnData object
        del gt_pca_adata.obsm['X_umap']
        del gt_pca_adata.uns['neighbors']
        logger.info("Done.")
        # set_trace()
        
        logger.info("Inverse transforming IFM generated ifm_pca_adata...")
        ifm_full_data = inverse_transform_gpu(ifm_pca_adata.X, pca)
        ifm_full_adata = ad.AnnData(X=ifm_full_data)
        ifm_full_adata.obs = ifm_pca_adata.obs.copy()
        ifm_full_adata.var_names = full_gt_adata.var_names
        
        # set_trace()
        logger.info("Done.")
        
        
        if args.plot_umap:
            if i == 0:
                logger.info("Plotting UMAP...")
                plot_umap(
                    gt_pca_data,
                    ifm_pca_data,
                    save_dir=umap_directory,
                    plot_name=f"calmflow_pp4_space{args.space_dim}",
                    labels=umap_labels
                )
                logger.info("Done.")
        if not args.compute_metrics:
            logger.info("NOT COMPUTING ANY METRICS!!")
            break
        else:
            logger.info("Computing metrics...")


            logger.info("----- Computing RBF-MMD and 2-Wass on 10D UMAP fit on PCA'd GT data... -----")
            umap_model_10D_on_GT = load_umap_model(f"{args.pert_split}_split_umap_model_10D.pkl")
            umap_10D_on_GT_ifm_data = umap_model_10D_on_GT.transform(ifm_pca_data)
            umap_10D_on_GT_gt_data = umap_model_10D_on_GT.transform(gt_pca_data)

            mmd = mmd_rbf(umap_10D_on_GT_ifm_data, umap_10D_on_GT_gt_data)
            umap10D_mmds.append(mmd)
            adaptive_mmd = mmd_rbf_adaptive(umap_10D_on_GT_ifm_data, umap_10D_on_GT_gt_data, k=args.adaptive_mmd_kernel_k)
            umap10D_adaptive_mmds.append(adaptive_mmd)
            wass = compute_wass(umap_10D_on_GT_ifm_data, umap_10D_on_GT_gt_data)
            umap10D_wasss.append(wass)
            
            logger.info(f"---> MMD: {mmd}")
            logger.info(f"---> Adaptive MMD: {adaptive_mmd}")
            logger.info(f"---> Wass: {wass}")
            logger.info("---------- Done. ----------")

            # logger.info(f"----- Computing RBF MMD and 2-Wass on the first {args.num_pca_dims} PCs of the PCA'd GT and IFM data... -----")
            # mmd = mmd_rbf(ifm_pca_data[:,:args.num_pca_dims], gt_pca_data[:,:args.num_pca_dims])
            # first_10_pc_mmds.append(mmd)
            # adaptive_mmd = mmd_rbf_adaptive(ifm_pca_data[:,:args.num_pca_dims], gt_pca_data[:,:args.num_pca_dims], k=args.adaptive_mmd_kernel_k)
            # first_10_pc_adaptive_mmds.append(adaptive_mmd)
            # # wass = compute_wass(ifm_pca_data[:,:args.num_pca_dims], gt_pca_data[:,:args.num_pca_dims])
            # # first_10_pc_wasss.append(wass)
            # logger.info(f"---> MMD: {mmd}")
            # logger.info(f"---> Adaptive MMD: {adaptive_mmd}")
            # # logger.info(f"---> Wass: {wass}")
            # logger.info("---------- Done. ----------")

            # logger.info(f"----- Computing RBF MMD and 2-Wass on the full 1000D PCA of the PCA'd GT and IFM data... -----")
            # mmd = mmd_rbf(ifm_pca_data, gt_pca_data)
            # full_pca_mmds.append(mmd)
            # adaptive_mmd = mmd_rbf_adaptive(ifm_pca_data, gt_pca_data, k=args.adaptive_mmd_kernel_k)
            # full_pca_adaptive_mmds.append(adaptive_mmd)
            # # wass = compute_wass(ifm_pca_data, gt_pca_data)
            # # full_pca_wasss.append(wass)
            # logger.info(f"---> MMD: {mmd}")
            # logger.info(f"---> Adaptive MMD: {adaptive_mmd}")
            # # logger.info(f"---> Wass: {wass}")
            # logger.info("---------- Done. ----------")

            logger.info(f"----- Computing RBF MMD and 2-Wass on 10D UMAP fit on PCA'd GT data and PCA'd IFM data combined... -----")
            combined_umap10D_gt_data, combined_umap10D_ifm_data = umap_embed(gt_pca_data, ifm_pca_data, 10)
            mmd = mmd_rbf(combined_umap10D_ifm_data, combined_umap10D_gt_data)
            umap_combined10D_mmds.append(mmd)
            adaptive_mmd = mmd_rbf_adaptive(combined_umap10D_ifm_data, combined_umap10D_gt_data, k=args.adaptive_mmd_kernel_k)
            umap_combined10D_adaptive_mmds.append(adaptive_mmd)
            wass = compute_wass(combined_umap10D_ifm_data, combined_umap10D_gt_data)
            umap_combined10D_wasss.append(wass)
            logger.info(f"---> MMD: {mmd}")
            logger.info(f"---> Adaptive MMD: {adaptive_mmd}")
            logger.info(f"---> Wass: {wass}")
            logger.info("---------- Done. ----------")
            
            if args.z_score:
                logger.info("Normalizing genes by Z-score...")
                ifm_full_cell_data = z_score_norm(ifm_full_cell_data)
                expression_data = z_score_norm(expression_data)
            
            # ----------------- TODO: need to take a closer look at this ----------------- #
            # Load UMAP model that is trained on PCA GT data
            umap_model_dir = "/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/umap_models"
            umap_model_path = os.path.join(umap_model_dir, f"{args.pert_split}_split_umap_model.pkl")
            with open(umap_model_path, 'rb') as f:
                logger.info(f"Loading UMAP model from {umap_model_path}...")
                umap_model = pickle.load(f)
                logger.info("Done.")
            
            
            logger.info("Transforming PCA expression data and PCA IFM data to 2D UMAP...")
            umap2D_gt_pca_data = umap_model.transform(gt_pca_data)
            umap2D_ifm_pca_data = umap_model.transform(ifm_pca_data)
            logger.info("Done.")

            full_gt_adata.obsm["X_umap"] = umap2D_gt_pca_data
            ifm_pca_adata.obsm["X_umap"] = umap2D_ifm_pca_data
            # Leiden clustering on PCA GT data
            
            with open(args.leiden_classifier_path, "r") as f:
                leiden_classifiers = json.load(f)
            leiden_classifier_weights = leiden_classifiers[args.pert_split][str(args.leiden_resol)]["weights"]
            num_leiden_classes =  leiden_classifiers[args.pert_split][str(args.leiden_resol)]["output_dim"]
            logger.info(f"Load Leiden classifier MLP from CHECKPOINT PATH: {leiden_classifier_weights}...")
            leiden_classifier_model = MLPClassifier(output_dim=num_leiden_classes)
            leiden_classifier_model.load_state_dict(torch.load(leiden_classifier_weights))
            leiden_classifier_model = leiden_classifier_model.to('cuda')
            leiden_classifier_model.eval()
            logger.info("Done.")

            logger.info("Predicting Leiden labels on generated ifm_pca_data...")
            X_gen = ifm_pca_adata.obsm['X_umap']
            X_gen_tensor = torch.tensor(X_gen, dtype=torch.float32, device="cuda")
            with torch.no_grad():
                outputs = leiden_classifier_model(X_gen_tensor)
                _, y_gen_pred = torch.max(outputs, 1)
            ifm_pca_adata.obs['leiden_pred'] = y_gen_pred.cpu().numpy().astype(str)
            
            logger.info("Plotting UMAP with predicted Leiden clusters for generated data...")
            if i == 0:
                sc.settings.figdir = umap_directory
                sc.pl.umap(ifm_pca_adata, color='leiden_pred', title='Generated Cells UMAP: Predicted Leiden Clusters', save='gen_data_Leiden_pred.png')
                sc.pl.umap(ifm_pca_adata, color='combined_labels', title='Generated Cells UMAP: Combo', save='gen_data_combined_labels.png')
            #TODO: make this line look better
            # full_gt_adata = sc.read_h5ad(f"/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/test_data_with_leiden_labels/{args.pert_split}/resol{args.leiden_resol}/test_adata_with_Leiden_resol{args.leiden_resol}_cluster{num_leiden_classes}.h5ad")
            # if i == 0:
            #     sc.pl.umap(full_gt_adata, color='leiden', title='PCA GT Data UMAP: Leiden Clusters', save='pca_gt_data_Leiden.png')
            #     sc.pl.umap(full_gt_adata, color='combined_labels', title='PCA GT Data UMAP: Combo', save='pca_gt_data_combined_labels.png')
            
            # # 4. compute inception score
            # predictions = leiden_classifier_model(X_gen_tensor).softmax(dim=1).detach()
            # incep_score = inception_score(predictions)
            # logger.info(f"---> Inception Score: {incep_score}")
            # # 5. compute leiden KL
            # leiden_kl_diveregnce = leiden_KL(full_gt_adata, ifm_pca_adata)
            # logger.info(f"--->Leiden KL Divergence: {leiden_kl_diveregnce}")
            # incep_scores.append(incep_score)
            # leiden_kls.append(leiden_kl_diveregnce)
            
            
            # assert umap2D_gt_pca_data.shape[1] == 2, f'Want 2D UMAP but got {umap2D_gt_pca_data.shape[1]}'
            # bin_edges = [
            #     np.linspace(np.min(umap2D_gt_pca_data[:, i]), np.max(umap2D_gt_pca_data[:, i]), args.kl_bins + 1)
            #     for i in range(umap2D_gt_pca_data.shape[1])
            # ]
            # logger.info(f"Created bins whose dimension 0 from {bin_edges[0][0]} to {bin_edges[0][-1]} and dimension 1 from {bin_edges[1][0]} to {bin_edges[1][-1]}..")
            
            # overall_binned_kl = binned_KL(bin_edges = bin_edges, gt_data=umap2D_gt_pca_data, gen_data=umap2D_ifm_pca_data, class_label="all")
            # overall_binned_kls.append(overall_binned_kl)
            # logger.info(f"---> Binned KL Divergence on all data: {overall_binned_kl}")
            # ------------------------------------------------------------------------- #

            # Compute metrics per class
            for class_label in label_combinations:
                logger.info(f"Class label: {class_label}")
                # Get data for this class
                # TODO: change to full_gt_adata:
                # ifm_full_cell_data, -> ifm_full_cell_adata Done
                #  expression_data, -> full_gt_adata, Done
                #  umap2D_gt_pca_data, -> full_gt_adata.obsm["X_umap"]
                #  umap2D_ifm_pca_data, -> ifm_pca_adata.obsm["X_umap"]
                # class_indices = test_dataset.get_indices_of_class(class_label)
                # class_ifm_full_cell_data = ifm_full_cell_data[class_indices]
                # class_expression_data = expression_data[class_indices] # or full_gt_adata[full_gt_adata.obs["combined_labels"]==class_label].X, equivalent
                # class_umap2D_gt_pca_data = umap2D_gt_pca_data[class_indices]
                # class_umap_cells = umap2D_ifm_pca_data[class_indices]

                class_ifm_full_adata = ifm_full_adata[ifm_full_adata.obs["combined_labels"]==class_label]
                class_full_gt_adata = full_gt_adata[full_gt_adata.obs["combined_labels"]==class_label]
                
                
                
                class_ifm_full_cell_data = ifm_full_adata[ifm_full_adata.obs["combined_labels"]==class_label].X
                class_expression_data = full_gt_adata[full_gt_adata.obs["combined_labels"]==class_label].X
                # class_umap2D_gt_pca_data = full_gt_adata.obsm["X_umap"][full_gt_adata.obs["combined_labels"]==class_label]
                # class_umap_cells = ifm_pca_adata.obsm["X_umap"][ifm_pca_adata.obs["combined_labels"]==class_label]
                # set_trace()


                # Find the control ifm_pca_data of the same cell type and chronicity
                # Subtract the mean control cell from the ifm_pca_data to get delta
                control_full_adata = get_control_adata(train_adata, class_label)
                mean_control_data_full = np.mean(control_full_adata.X, axis=0) # shape (21710,)
                
                # Full Cell Data
                logger.info(f"Full {class_ifm_full_cell_data.shape[1]} dimensional Cell Data")
                r2, pearson_corr, spearman_corr = compute_statistics(class_ifm_full_cell_data, class_expression_data)
                delta_r2, delta_pearson_corr, delta_spearman_corr = compute_statistics(class_ifm_full_cell_data - mean_control_data_full, class_expression_data - mean_control_data_full)
                # kl_divergence = binned_KL(bin_edges = bin_edges, gt_data=class_umap2D_gt_pca_data, gen_data=class_umap_cells, class_label=class_label)
                logger.info(f"---> R^2: {str(f'{r2:.6f}').ljust(10)} "
                    f"---> Pearson correlation: {str(f'{pearson_corr:.6f}').ljust(10)} "
                    f"---> Spearman correlation: {str(f'{spearman_corr:.6f}').ljust(10)}"
                    f"---> Delta IFM R^2: {str(f'{delta_r2:.6f}').ljust(10)} "
                    f"---> Delta Pearson correlation: {str(f'{delta_pearson_corr:.6f}').ljust(10)} "
                    f"---> Delta Spearman correlation: {str(f'{delta_spearman_corr:.6f}').ljust(10)}"
                    # f"Binned 2D UMAP KL Divergence: {str(f'{kl_divergence:.6f}').ljust(10)} "
                    )
                # set_trace()
                ifm_r2s[class_label].append(r2)
                ifm_pears[class_label].append(pearson_corr)
                ifm_spears[class_label].append(spearman_corr)
                
                ifm_r2s_delta[class_label].append(delta_r2)
                ifm_pears_delta[class_label].append(delta_pearson_corr)
                ifm_spears_delta[class_label].append(delta_spearman_corr)
                # binned_kls[class_label].append(kl_divergence)


                # Compute the top 50 differentially expressed genes using scanpy between control_full_adata and class_test_control_combined_adata
                class_test_control_combined_adata = full_gt_adata[full_gt_adata.obs["combined_labels"] == class_label]
                class_test_control_combined_adata = class_test_control_combined_adata.concatenate(control_full_adata)
                sc.tl.rank_genes_groups(class_test_control_combined_adata, groupby='combined_labels', n_genes=50, method='t-test')
                de_genes = [gene for pair in class_test_control_combined_adata.uns['rank_genes_groups']['names'] for gene in pair]
                # class_test_control_combined_de_adata = class_test_control_combined_adata[:, de_genes]
                class_ifm_full_adata_de = class_ifm_full_adata[:, de_genes]
                class_full_gt_adata_de = class_full_gt_adata[:, de_genes]
                control_full_adata_de = control_full_adata[:, de_genes]
                mean_control_data_de = np.mean(control_full_adata_de.X, axis=0)

                # set_trace()
                logger.info(f"Top {len(de_genes)} DE genes")
                r2, pearson_corr, spearman_corr = compute_statistics(class_ifm_full_adata_de.X, class_full_gt_adata_de.X)
                delta_r2, delta_pearson_corr, delta_spearman_corr = compute_statistics(class_ifm_full_adata_de.X - mean_control_data_de, class_full_gt_adata_de.X - mean_control_data_de)
                logger.info(
                    f"---> DE GENES R^2: {str(f'{r2:.6f}').ljust(10)} "
                    f"---> Pearson correlation: {str(f'{pearson_corr:.6f}').ljust(10)} "
                    f"---> Spearman correlation: {str(f'{spearman_corr:.6f}').ljust(10)} "
                    f"---> Delta IFM DE GENES R^2: {str(f'{delta_r2:.6f}').ljust(10)} "
                    f"---> Delta Pearson correlation: {str(f'{delta_pearson_corr:.6f}').ljust(10)} "
                    f"Delta Spearman correlation: {str(f'{delta_spearman_corr:.6f}').ljust(10)} ")
                ifm_r2s_de[class_label].append(r2)
                ifm_pears_de[class_label].append(pearson_corr)
                ifm_spears_de[class_label].append(spearman_corr)
                
                ifm_r2s_de_delta[class_label].append(delta_r2)
                ifm_pears_de_delta[class_label].append(delta_pearson_corr)
                ifm_spears_de_delta[class_label].append(delta_spearman_corr)

            logger.info(f"\nTemperature {args.temp}")

                
    
    umap10D_mmds = np.array(umap10D_mmds)
    umap10D_adaptive_mmds = np.array(umap10D_adaptive_mmds)
    umap10D_wasss = np.array(umap10D_wasss)
    
    umap_combined10D_mmds = np.array(umap_combined10D_mmds)
    umap_combined10D_adaptive_mmds = np.array(umap_combined10D_adaptive_mmds)
    umap_combined10D_wasss = np.array(umap_combined10D_wasss)
    
    # full_pca_mmds = np.array(full_pca_mmds)
    # full_pca_adaptive_mmds = np.array(full_pca_adaptive_mmds)
    # full_pca_wasss = np.array(full_pca_wasss)
    
    # first_10_pc_mmds = np.array(first_10_pc_mmds)
    # first_10_pc_adaptive_mmds = np.array(first_10_pc_adaptive_mmds)
    # # first_10_pc_wasss = []
        
    
    # overall_binned_kls = np.array(overall_binned_kl)
    
    # logger.info(f"First 10 PCs: MMD Mean {first_10_pc_mmds.mean():.6f} STD {first_10_pc_mmds.std():.6f}")
    # logger.info(f"First 10 PCs: Adaptive MMD Mean {first_10_pc_adaptive_mmds.mean():.6f} STD {first_10_pc_adaptive_mmds.std():.6f}")
    # logger.info(f"Full PCA: MMD Mean {full_pca_mmds.mean():.6f} STD {full_pca_mmds.std():.6f}")
    # logger.info(f"Full PCA: Adaptive MMD Mean {full_pca_adaptive_mmds.mean():.6f} STD {full_pca_adaptive_mmds.std():.6f}")
    # logger.info(f"2-Wasserstein Mean {wasss.mean():.6f} STD {wasss.std():.6f}\n")
    # logger.info(f"Overall binned KL Mean {overall_binned_kls.mean():.6f} STD {overall_binned_kls.std():.6f}\n")
         
    # binned_kls = {class_label: np.array(binned_kls[class_label]) for class_label in binned_kls}     
    
    ifm_r2s = {class_label: np.array(ifm_r2s[class_label]) for class_label in ifm_r2s}
    ifm_pears = {class_label: np.array(ifm_pears[class_label]) for class_label in ifm_pears}
    ifm_spears = {class_label: np.array(ifm_spears[class_label]) for class_label in ifm_spears}
    
    ifm_r2s_de = {class_label: np.array(ifm_r2s_de[class_label]) for class_label in ifm_r2s_de}
    ifm_pears_de = {class_label: np.array(ifm_pears_de[class_label]) for class_label in ifm_pears_de}
    ifm_spears_de = {class_label: np.array(ifm_spears_de[class_label]) for class_label in ifm_spears_de}

    ifm_r2s_delta = {class_label: np.array(ifm_r2s_delta[class_label]) for class_label in ifm_r2s_delta}
    ifm_pears_delta = {class_label: np.array(ifm_pears_delta[class_label]) for class_label in ifm_pears_delta}
    ifm_spears_delta = {class_label: np.array(ifm_spears_delta[class_label]) for class_label in ifm_spears_delta}

    ifm_r2s_de_delta = {class_label: np.array(ifm_r2s_de_delta[class_label]) for class_label in ifm_r2s_de_delta}
    ifm_pears_de_delta = {class_label: np.array(ifm_pears_de_delta[class_label]) for class_label in ifm_pears_de_delta}
    ifm_spears_de_delta = {class_label: np.array(ifm_spears_de_delta[class_label]) for class_label in ifm_spears_de_delta}

    # for class_label in label_combinations:
    #     logger.info(f"Class {class_label} IFM R^2 {ifm_r2s[class_label].mean():.6f} +/- {ifm_r2s[class_label].std():.6f}, Pearson {ifm_pears[class_label].mean():.6f} +/- {ifm_pears[class_label].std():.6f}, Spearman {ifm_spears[class_label].mean():.6f} +/- {ifm_spears[class_label].std():.6f}") #, KL {binned_kls[class_label].mean():.6f} +/- {binned_kls[class_label].std():.6f}")
    #     logger.info(f"Class {class_label} IFM DE R^2 {ifm_r2s_de[class_label].mean():.6f} +/- {ifm_r2s_de[class_label].std():.6f}, Pearson {ifm_pears_de[class_label].mean():.6f} +/- {ifm_pears_de[class_label].std():.6f}, Spearman {ifm_spears_de[class_label].mean():.6f} +/- {ifm_spears_de[class_label].std():.6f}")
    #     logger.info(f"Class {class_label} Delta IFM R^2 {ifm_r2s_delta[class_label].mean():.6f} +/- {ifm_r2s_delta[class_label].std():.6f}, Delta Pearson {ifm_pears_delta[class_label].mean():.6f} +/- {ifm_pears_delta[class_label].std():.6f}, Delta Spearman {ifm_spears_delta[class_label].mean():.6f} +/- {ifm_spears_delta[class_label].std():.6f}")
    #     logger.info(f"Class {class_label} Delta IFM DE R^2 {ifm_r2s_de_delta[class_label].mean():.6f} +/- {ifm_r2s_de_delta[class_label].std():.6f}, Delta Pearson {ifm_pears_de_delta[class_label].mean():.6f} +/- {ifm_pears_de_delta[class_label].std():.6f}, Delta Spearman {ifm_spears_de_delta[class_label].mean():.6f} +/- {ifm_spears_de_delta[class_label].std():.6f}")
    
    # Calculate overall mean and std for R^2, Pearson, and Spearman correlations
    all_r2s = np.concatenate([ifm_r2s[label] for label in label_combinations])
    all_pears = np.concatenate([ifm_pears[label] for label in label_combinations])
    all_spears = np.concatenate([ifm_spears[label] for label in label_combinations])
    
    all_r2s_de = np.concatenate([ifm_r2s_de[label] for label in label_combinations])
    all_pears_de = np.concatenate([ifm_pears_de[label] for label in label_combinations])
    all_spears_de = np.concatenate([ifm_spears_de[label] for label in label_combinations])
    
    all_r2s_delta = np.concatenate([ifm_r2s_delta[label] for label in label_combinations])
    all_pears_delta = np.concatenate([ifm_pears_delta[label] for label in label_combinations])
    all_spears_delta = np.concatenate([ifm_spears_delta[label] for label in label_combinations])
    
    all_r2s_de_delta = np.concatenate([ifm_r2s_de_delta[label] for label in label_combinations])
    all_pears_de_delta = np.concatenate([ifm_pears_de_delta[label] for label in label_combinations])
    all_spears_de_delta = np.concatenate([ifm_spears_de_delta[label] for label in label_combinations])
    
    logger.info(f"MMD GT UMAP: Mean {umap10D_mmds.mean():.6f} STD {umap10D_mmds.std():.6f}")
    logger.info(f"Wasserstein GT UMAP: Mean {umap10D_wasss.mean():.6f} STD {umap10D_wasss.std():.6f}")
    logger.info(f"MMD UMAP Together: Mean {umap_combined10D_mmds.mean():.6f} STD {umap_combined10D_mmds.std():.6f}")
    logger.info(f"Wasserstein UMAP Together: Mean {umap_combined10D_wasss.mean():.6f} STD {umap_combined10D_wasss.std():.6f}")
    logger.info(f"Leiden KL Divergence: placeholder")
    logger.info(f"Inception Score: placeholder")
    logger.info(f"Overall IFM R^2: Mean {all_r2s.mean():.6f} STD {all_r2s.std():.6f}")
    logger.info(f"Overall IFM Pearson: Mean {all_pears.mean():.6f} STD {all_pears.std():.6f}")
    logger.info(f"Overall IFM Spearman: Mean {all_spears.mean():.6f} STD {all_spears.std():.6f}")
    
    logger.info(f"Overall IFM TOP {len(de_genes)} DE GENES R^2: Mean {all_r2s_de.mean():.6f} STD {all_r2s_de.std():.6f}")
    logger.info(f"Overall IFM TOP {len(de_genes)} DE GENES Pearson: Mean {all_pears_de.mean():.6f} STD {all_pears_de.std():.6f}")
    logger.info(f"Overall IFM TOP {len(de_genes)} DE GENES Spearman: Mean {all_spears_de.mean():.6f} STD {all_spears_de.std():.6f}")
    
    logger.info(f"Overall Delta IFM R^2: Mean {all_r2s_delta.mean():.6f} STD {all_r2s_delta.std():.6f}")
    logger.info(f"Overall Delta IFM Pearson: Mean {all_pears_delta.mean():.6f} STD {all_pears_delta.std():.6f}")
    logger.info(f"Overall Delta IFM Spearman: Mean {all_spears_delta.mean():.6f} STD {all_spears_delta.std():.6f}")
    
    logger.info(f"Overall Delta IFM TOP {len(de_genes)} DE GENES R^2: Mean {all_r2s_de_delta.mean():.6f} STD {all_r2s_de_delta.std():.6f}")
    logger.info(f"Overall Delta IFM TOP {len(de_genes)} DE GENES Pearson: Mean {all_pears_de_delta.mean():.6f} STD {all_pears_de_delta.std():.6f}")
    logger.info(f"Overall Delta IFM TOP {len(de_genes)} DE GENES Spearman: Mean {all_spears_de_delta.mean():.6f} STD {all_spears_de_delta.std():.6f}")
    
    logger.info(f"Adaptive MMD (k={args.adaptive_mmd_kernel_k}) GT UMAP: Mean {umap10D_adaptive_mmds.mean():.6f} STD {umap10D_adaptive_mmds.std():.6f}")
    logger.info(f"Adaptive MMD UMAP Together (k={args.adaptive_mmd_kernel_k}) Mean {umap_combined10D_adaptive_mmds.mean():.6f} STD {umap_combined10D_adaptive_mmds.std():.6f}")
    
    # Log mean and std of leiden kl and incep scores
    # leiden_kl_mean = np.mean(leiden_kls)
    # leiden_kl_std = np.std(leiden_kls)
    # # set_trace()
    # incep_score_mean = np.mean(incep_scores)
    # incep_score_std = np.std(incep_scores)
    
    # logger.info("Preparing metrics to a pandas dataframe...")
    # # Prepare the data for the DataFrame
    # data = []

    # # Add MMD and Wasserstein metrics
    # data.append(["Overall", "MMD", mmds.tolist()])
    # data.append(["Overall", "2-Wasserstein", wasss.tolist()])
    # data.append(["Overall", "Leiden KL", leiden_kls])
    # data.append(["Overall", "Inception Score", incep_scores])
    # data.append(["Overall", "Binned KL", overall_binned_kls.tolist()])

    # # Add IFM metrics for each class label
    # for class_label in label_combinations:
    #     data.append([class_label, "IFM R^2", ifm_r2s[class_label].tolist()])
    #     data.append([class_label, "IFM Pearson", ifm_pears[class_label].tolist()])
    #     data.append([class_label, "IFM Spearman", ifm_spears[class_label].tolist()])
    #     data.append([class_label, "Delta IFM R^2", ifm_r2s_delta[class_label].tolist()])
    #     data.append([class_label, "Delta IFM Pearson", ifm_pears_delta[class_label].tolist()])
    #     data.append([class_label, "Delta IFM Spearman", ifm_spears_delta[class_label].tolist()])
    # # Create a DataFrame
    # df = pd.DataFrame(data, columns=["Class Label", "Metric", "Values"])
    # logger.info("Done.")

    # # set_trace()
    # # Define the CSV file path
    # csv_file_path = os.path.join(run_directory, "computed_metrics.csv")
    # logger.info(f"Saving computed metrics to {csv_file_path}")
    # # Save the DataFrame to a CSV file
    # df.to_csv(csv_file_path, index=False)
    # logger.info("Done.")
    
    

    # logger.info(f"Leiden KL Divergence Mean: {leiden_kl_mean:.6f} +/- {leiden_kl_std:.6f}")
    # logger.info(f"Inception Score Mean: {incep_score_mean:.6f} +/- {incep_score_std:.6f}")

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
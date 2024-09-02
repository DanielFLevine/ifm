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

from utils.modules import CustomVAEDecoder, TwoLayerMLP
from utils.metrics import mmd_rbf, compute_wass, transform_gpu, umap_embed
from utils.plots import plot_umap

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cp_dir",
        type=str,
        default="/gpfs/radev/scratch/dijk/sh2748/calmflow_singlecell/training_runs" # TODO: change to my directory
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
        default="pythia-160m-timepoints16-straightpathTrue-drop0.0ifm-2024-06-06_00-39-44" # TODO: maybe change this
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


def main(args):
    # Prep data
    dataset_path = "/home/dfl32/project/ifm/cinemaot_data/conditional_cinemaot/ct_pert_split"
    test_dataset = load_from_disk(os.path.join(dataset_path, 'test_ds'))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    
    expression_matrix = np.array(test_dataset['expression'])
    adata = ad.AnnData(X=expression_matrix)
    # adata = sc.read_h5ad("/home/dfl32/project/ifm/cinemaot_data/raw_cinemaot.h5ad")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    # Don't have the following keys so removed these
    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    # adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    # The data seems to already have been preprocessed so remove these
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=args.hvgs)

    hvgs = adata.var['highly_variable']
    rare_hvgs = hvgs & (adata.var['n_cells'] < args.n_cell_thresh)

    if issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X
    # set_trace()
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

    ### IFM ###
    # Load IFM model
    with open(args.model_json_path, "r") as f:
        model_paths = json.load(f)
    # set_trace()
    weights = "pretrained_weights" if args.pretrained_weights else "random_weights"
    cp_path = model_paths[weights][str(args.space_dim)]
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
    # set_trace()
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
    time_points = args.time_points

    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    if args.idfm:
        euler_step_size = 1/(time_points-1)
    for i in range(num_repeats):
        with torch.no_grad():
            cells = []
            umap_labels = {
                "perturbation": [],
                "cell_type": [],
                "chronicity": []
            }
            for batch in tqdm(test_dataloader): # change to data loader
                # Generate a batch of cells
                
                # Prepare labels
                umap_labels["perturbation"].extend(batch["perturbation"])
                umap_labels["cell_type"].extend(batch["cell_type"])
                umap_labels["chronicity"].extend(batch["chronicity"])
                
            #     prefix_tokens = build_and_tokenize_prefixes(batch, prompts, tokenizer)
            #     prefix_input_ids = prefix_tokens['prefix_input_ids']
            #     prefix_tokens_attention_mask = prefix_tokens['prefix_attention_mask']
            #     prefix_length = prefix_input_ids.shape[1]
            #     batch_size = prefix_input_ids.shape[0]
            #     paths = torch.normal(0.0, 1.0, size=(batch_size, args.points_per_sample, input_dim)).to(device)
            #     prefix_emb = model.gpt_neox.embed_in(prefix_input_ids)
                
            #     # Autoregressive generation
            #     for _ in range(time_points-1):
            #         path_emb = model.cell_enc(paths)
            #         # Reshape for spatial integration
            #         batch_size, seq_len, feature = path_emb.shape
            #         path_emb = path_emb.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
            #         path_emb = path_emb.view(batch_size, seq_len* args.space_dim, feature // args.space_dim)
            #         # set_trace()
            #         # print(prefix_emb.shape, path_emb.shape)
            #         input_emb = torch.concat([prefix_emb, path_emb], dim=1)
            #         attention_mask = torch.cat(
            #                 [
            #                     prefix_tokens_attention_mask,
            #                     torch.ones((path_emb.shape[0], path_emb.shape[1]), dtype=torch.int32).to(model.device)
            #                 ], 
            #                 dim=1
            #         )
                    
            #         outputs = model.gpt_neox(inputs_embeds=input_emb, attention_mask=attention_mask).last_hidden_state
            #         outputs = outputs[:, prefix_length:, ...]
                    
            #         if not args.reshape_postvae:
            #             outputs = outputs.view(batch_size, seq_len, args.space_dim, feature // args.space_dim)
            #             outputs = outputs.view(batch_size, seq_len, feature)
                    
            #         outputs, _, _ = model.cell_dec(outputs, temperature=args.temp)
            #         last_outputs = outputs[:, -args.points_per_sample:, :]
                    
            #         if args.idfm:
            #             last_outputs = paths[:, -args.points_per_sample:, :] + (euler_step_size * last_outputs)
                    
            #         paths = torch.concat([paths, last_outputs], axis=1)
                    
                
            #     batch_size, _, feature_dim = outputs.shape
            #     cells.append(outputs[:, -args.points_per_sample:, :].reshape(args.points_per_sample*batch_size, feature_dim).detach().cpu().numpy())
            # cells = np.concatenate(cells, axis=0)
        # set_trace()
        cells = np.load("generated_cells.npy")
        logger.info("Inverse transforming IFM generated cells...")
        cells_ag = inverse_transform_gpu(cells, pca)
        # set_trace()
        # cells_ag = pca.inverse_transform(cells)
        logger.info("Done.")
        
        sample_indices = np.random.choice(expression_data.shape[0], size=num_samples*args.points_per_sample, replace=False)
        # sampled_expression_data = expression_data[sample_indices]
        # BUG: WARNING: the following may be non-sense
        logger.info("Inverse transforming GT cells...")
        pca_sampled_expression_data = expression_data[sample_indices]
        sampled_expression_data = inverse_transform_gpu(pca_sampled_expression_data, pca)
        logger.info("Done.")
        # set_trace()
        # logger.info("PCAing ground truth data...")
        # pca_sampled_expression_data = transform_gpu(sampled_expression_data, pca)
        # logger.info("Done.")

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        umap_directory = os.path.join("/home/sh2748/ifm/umaps", current_time)
        os.makedirs(umap_directory, exist_ok=True)
        umap_name = os.path.join
        if args.plot_umap:
            if i == 0:
                logger.info("Plotting UMAP...")
                plot_umap(
                    pca_sampled_expression_data,
                    cells,
                    plot_name=f"{current_time}/calmflow_pp4_space{args.space_dim}",
                    labels=umap_labels
                )

        if args.umap_embed:
            pca_sampled_expression_data, cells = umap_embed(pca_sampled_expression_data, cells)
        mmd = mmd_rbf(cells[:,:args.num_pca_dims], pca_sampled_expression_data[:,:args.num_pca_dims])
        mmds.append(mmd)
        logger.info(f"MMD: {mmd}")
        wass = compute_wass(cells[:,:args.num_pca_dims], pca_sampled_expression_data[:,:args.num_pca_dims])
        wasss.append(wass)
        logger.info(f"Wass: {wass}")

        if args.z_score:
            logger.info("Normalizing genes by Z-score...")
            cells_ag = z_score_norm(cells_ag)
            sampled_expression_data = z_score_norm(sampled_expression_data)

        logger.info("Computing metrics...")
    #     # All genes
    #     # BUG: cell_ag is has feature dimension 21710 or something because it's inverse-pca'd but sampled_expression_data is not
        r2, pearson_corr, spearman_corr = compute_statistics(cells_ag, sampled_expression_data)
        logger.info(f"IFM R^2: {r2}")
        logger.info(f"IFM Pearson correlation: {pearson_corr}")
        logger.info(f"IFM Spearman correlation: {spearman_corr}")
        ifm_r2s.append(r2)
        ifm_pears.append(pearson_corr)
        ifm_spears.append(spearman_corr)

    #     # HVGS
    #     set_trace()
    #     r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, hvgs], sampled_expression_data[:, hvgs])
    #     logger.info(f"IFM HVGS R^2: {r2}")
    #     logger.info(f"IFM HVGS Pearson correlation: {pearson_corr}")
    #     logger.info(f"IFM HVGS Spearman correlation: {spearman_corr}")
    #     ifm_r2s_hvg.append(r2)
    #     ifm_pears_hvg.append(pearson_corr)
    #     ifm_spears_hvg.append(spearman_corr)

    #     # Rare HVGS
    #     r2, pearson_corr, spearman_corr = compute_statistics(cells_ag[:, rare_hvgs], sampled_expression_data[:, rare_hvgs])
    #     logger.info(f"IFM Rare HVGS R^2: {r2}")
    #     logger.info(f"IFM Rare HVGS Pearson correlation: {pearson_corr}")
    #     logger.info(f"IFM Rare HVGS Spearman correlation: {spearman_corr}")
    #     ifm_r2s_hvg_rare.append(r2)
    #     ifm_pears_hvg_rare.append(pearson_corr)
    #     ifm_spears_hvg_rare.append(spearman_corr)

    logger.info(f"\nTemperature {args.temp}")
    ifm_r2s = np.array(ifm_r2s)
    ifm_pears = np.array(ifm_pears)
    ifm_spears = np.array(ifm_spears)
    mmds = np.array(mmds)
    wasss = np.array(wasss)
    logger.info(f"MMD Mean {mmds.mean()} STD {mmds.std()}")
    logger.info(f"2-Wasserstein Mean {wasss.mean()} STD {wasss.std()}\n")

    logger.info(f"IFM R^2 Mean {ifm_r2s.mean()} STD {ifm_r2s.std()}")
    logger.info(f"IFM Pearson Mean {ifm_pears.mean()} STD {ifm_pears.std()}")
    logger.info(f"IFM Spearman Mean {ifm_spears.mean()} STD {ifm_spears.std()}")

    # ifm_r2s_hvg = np.array(ifm_r2s_hvg)
    # ifm_pears_hvg = np.array(ifm_pears_hvg)
    # ifm_spears_hvg = np.array(ifm_spears_hvg)
    # logger.info(f"IFM HVGS R^2 Mean {ifm_r2s_hvg.mean()} STD {ifm_r2s_hvg.std()}")
    # logger.info(f"IFM HVGS Pearson Mean {ifm_pears_hvg.mean()} STD {ifm_pears_hvg.std()}")
    # logger.info(f"IFM HVGS Spearman Mean {ifm_spears_hvg.mean()} STD {ifm_spears_hvg.std()}")

    # ifm_r2s_hvg_rare = np.array(ifm_r2s_hvg_rare)
    # ifm_pears_hvg_rare = np.array(ifm_pears_hvg_rare)
    # ifm_spears_hvg_rare = np.array(ifm_spears_hvg_rare)
    # logger.info(f"IFM Rare HVGS R^2 Mean {ifm_r2s_hvg_rare.mean()} STD {ifm_r2s_hvg_rare.std()}")
    # logger.info(f"IFM Rare HVGS Pearson Mean {ifm_pears_hvg_rare.mean()} STD {ifm_pears_hvg_rare.std()}")
    # logger.info(f"IFM Rare HVGS Spearman Mean {ifm_spears_hvg_rare.mean()} STD {ifm_spears_hvg_rare.std()}")

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
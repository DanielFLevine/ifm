import argparse
import json
import logging
import os
import pickle
import random
from datetime import datetime
from tqdm import tqdm

from accelerate import Accelerator
import anndata
import numpy as np
import safetensors
import scanpy as sc
import scvi
import torch
import wandb
from datasets import load_from_disk
from torch import nn
from torch.distributions import Normal
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils import cpp_extension
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    EarlyStoppingCallback,
    TrainingArguments,
    GPTNeoXForCausalLM,
    GPTNeoXConfig
)
from optimum.bettertransformer import BetterTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.combo_split import combo_split_nochron
from utils.custom_trainer import LLMVAETrainer
from utils.modules import CustomDecoder, CustomVAEDecoder, CustomSCVIDecoder, TwoLayerMLP

from torch.nn.parallel import DistributedDataParallel as DDP

import umap
import matplotlib.pyplot as plt

logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/home/dfl32/project/huggingface"
if "LOCAL_RANK" in os.environ:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
else:
    LOCAL_RANK = 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/dfl32/scratch/cinemaot_data/cell_sentences_hf/jeff_data_HF_dataset_dict"
    )
    parser.add_argument(
        "--adata_file_path",
        type=str,
        default="/home/dfl32/project/ifm/cinemaot_data/hvg_normalized_cinemaot.h5ad"
    )
    parser.add_argument(
        "--llm_dataset_path",
        type=str,
        default="/home/dfl32/scratch/cinemaot_data/ifm_hf_ds/gaussian_pca768_normFalse_hf_ds"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="/home/dfl32/project/ifm/prompts/cinemaot_prompts.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/dfl32/scratch/training-runs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--scvi_checkpoint",
        type=str,
        default="/home/dfl32/project/ifm/scvi_models/epoch99_layers3_latent30_hidden128model.pt"
    )
    parser.add_argument(
        "--time_points",
        type=int,
        default=16
    )
    parser.add_argument(
        "--path_var",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=50
    )
    parser.add_argument(
        "--max_num_blocks",
        type=int,
        default=32
    )
    parser.add_argument(
        "--use_flash_attention_2",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=None
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="IFM"
    )
    parser.add_argument(
        "--wandb_logging",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--wandb_run_base_name",
        type=str,
        default="ifm"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=10
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=10
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=5
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=100
    )
    parser.add_argument(
        "--normalize_output",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--e2e",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--train_gaussian",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--use_vae",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--straight_paths",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--skip_conn_dec",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--target_dist",
        type=str,
        default=None
    )
    parser.add_argument(
        "--target_dim",
        type=int,
        default=5000
    )
    parser.add_argument(
        "--just_llm",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--train_custom",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--hdim_2d",
        type=int,
        default=32
    )
    parser.add_argument(
        "--idim_2d",
        type=int,
        default=32
    )
    parser.add_argument(
        "--nheads_2d",
        type=int,
        default=4
    )
    parser.add_argument(
        "--nblocks_2d",
        type=int,
        default=1
    )
    parser.add_argument(
        "--ifm_reg",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--sigma_decay",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.0001
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--scvi_dec",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--time_scale",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--scale_last",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--scale_last_weight",
        type=int,
        default=10
    )
    parser.add_argument(
        "--space_dim",
        type=int,
        default=1
    )
    parser.add_argument(
        "--reshape_postvae",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--mlp_enc",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--use_pretrained",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--idfm",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--mlp_musig",
        type=bool,
        default=False
    )
    return parser.parse_args()


def generate_paths(X_0, X_1, sigma=0.0001, time_points=16, straight_paths=False):
    # Convert lists to tensors
    X_0 = torch.tensor(X_0, dtype=torch.float32)
    X_1 = torch.tensor(X_1, dtype=torch.float32)
    
    # Dimensions
    dim = len(X_0)  # Here dim is 5000
    
    # Generate time points: from 0 to 1, including both endpoints, evenly spaced
    times = torch.linspace(0, 1, steps=time_points).view(time_points, 1)
    
    # Expand times and inputs to broadcast across dimensions
    times_expanded = times.expand(time_points, dim)
    
    # Linear interpolation: tX_1 + (1-t)X_0 = X_0 + t(X_1 - X_0)
    path_means = X_0 + times_expanded * (X_1 - X_0)
    
    # Initialize paths with means (ensures exact start at X_0 and end at X_1)
    paths = path_means.clone()
    
    # Gaussian noise: zero mean, sigma standard deviation, but not for the first and last time points
    if straight_paths:
        return paths
    
    if time_points > 2:
        noise = sigma * torch.randn(time_points-2, dim)
        
        # Determine where X_0 or X_1 is non-zero, for intermediate time points
        non_zero_mask = ((X_0 != 0) | (X_1 != 0))
        non_zero_mask_expanded = non_zero_mask.unsqueeze(0).expand(time_points-2, -1)
        
        # Apply noise only where non_zero_mask is True, and only to intermediate points
        paths[1:-1] = paths[1:-1].where(~non_zero_mask_expanded, paths[1:-1] + noise)

    return paths

def plot_umap(
    gt_pca,
    model,
    time_points,
    input_dim,
    space_dim,
    device,
    temp=1.0,
    num_samples=5000,
    batch_size=100,
    idfm=False
):
    model.eval()
    num_steps = num_samples//batch_size

    euler_step_size = 1/(time_points-1)
    all_trajs = []
    with torch.no_grad():
        cells = []
        for step in tqdm(range(num_steps)):
            inputs = torch.normal(0.0, 1.0, size=(batch_size, 1, input_dim)).to(device)
            for _ in range(time_points-1):
                outputs = model.cell_enc(inputs)

                # Reshape for spatial integration
                batch_size, seq_len, feature = outputs.shape
                outputs = outputs.view(batch_size, seq_len, space_dim, feature // space_dim)
                outputs = outputs.view(batch_size, seq_len* space_dim, feature // space_dim)


                outputs = model.gpt_neox(inputs_embeds=outputs).last_hidden_state

                outputs, _, _ = model.cell_dec(outputs, temperature=temp)
                last_outputs = outputs[:, -1:, :]
                if idfm:
                    last_outputs = inputs[:, -1:, :] + (euler_step_size * last_outputs)
                inputs = torch.concat([inputs, last_outputs], axis=1)
            cells.append(outputs[:, -1, :].detach().cpu().numpy())
            all_trajs.append(inputs.detach().cpu().numpy())
        cells = np.concatenate(cells, axis=0)
        all_trajs = np.concatenate(all_trajs, axis=0)
    
    # Combine the two datasets
    combined_data = np.vstack((gt_pca, cells))

    # Fit and transform the combined data using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_model.fit_transform(combined_data)

    # Split the transformed data back into the two original sets
    umap_gt_pca = umap_embedding[:num_samples]
    umap_cells = umap_embedding[num_samples:]

    # Plot the results
    plt.figure(figsize=(5, 4))
    plt.scatter(umap_gt_pca[:, 0], umap_gt_pca[:, 1], color='blue', label='ground truth', alpha=0.5, s=0.5)
    plt.scatter(umap_cells[:, 0], umap_cells[:, 1], color='red', label='generated', alpha=0.5, s=0.5)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{time_points} Time Points\n{input_dim} PCA Dimensions\n{num_samples} Cells')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

def copy_weights(pretrained_model, custom_model, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size):
    # Copy the weights for each relevant layer
    for layer_idx in range(num_hidden_layers):
        # Access the pretrained layer
        pretrained_layer = pretrained_model.gpt_neox.layers[layer_idx]
        custom_layer = custom_model.gpt_neox.layers[layer_idx]

        # Copy attention weights from query_key_value
        pretrained_qkv = pretrained_layer.attention.query_key_value.weight.data
        custom_qkv = custom_layer.attention.query_key_value.weight.data

        # Slice the pretrained weights to fit the custom model's dimensions
        custom_qkv.copy_(pretrained_qkv[:custom_qkv.shape[0], :custom_qkv.shape[1]])
        
        pretrained_qkv_bias = pretrained_layer.attention.query_key_value.bias.data
        custom_qkv_bias = custom_layer.attention.query_key_value.bias.data
        custom_qkv_bias.copy_(pretrained_qkv_bias[:custom_qkv_bias.shape[0]])

        # Copy attention dense weights
        custom_dense_weight = custom_layer.attention.dense.weight.data
        pretrained_dense_weight = pretrained_layer.attention.dense.weight.data
        custom_dense_weight.copy_(pretrained_dense_weight[:custom_dense_weight.shape[0], :custom_dense_weight.shape[1]])
        
        custom_layer.attention.dense.bias.data.copy_(pretrained_layer.attention.dense.bias.data[:hidden_size])

        # Copy dense weights
        custom_layer.mlp.dense_h_to_4h.weight.data.copy_(pretrained_layer.mlp.dense_h_to_4h.weight.data[:intermediate_size, :hidden_size])
        custom_layer.mlp.dense_h_to_4h.bias.data.copy_(pretrained_layer.mlp.dense_h_to_4h.bias.data[:intermediate_size])
        
        custom_layer.mlp.dense_4h_to_h.weight.data.copy_(pretrained_layer.mlp.dense_4h_to_h.weight.data[:hidden_size, :intermediate_size])
        custom_layer.mlp.dense_4h_to_h.bias.data.copy_(pretrained_layer.mlp.dense_4h_to_h.bias.data[:hidden_size])

    # Copy the final layer norm weights
    custom_model.gpt_neox.final_layer_norm.weight.data.copy_(pretrained_model.gpt_neox.final_layer_norm.weight.data[:hidden_size])
    custom_model.gpt_neox.final_layer_norm.bias.data.copy_(pretrained_model.gpt_neox.final_layer_norm.bias.data[:hidden_size])


def main(args):

    logger.info(f"\nLOCAL RANK: {LOCAL_RANK}")
    logger.info(f"CUDA HOME: {cpp_extension.CUDA_HOME}")
    logger.info(f"\nTORCH CUDA VERSION: {torch.version.cuda}")

    dataset = load_from_disk(args.llm_dataset_path)
    if args.train_dataset_size:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(args.train_dataset_size))

    split_dataset = dataset.train_test_split(
        test_size=0.1, 
        shuffle=True, 
        seed=42
    )

    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    val_dataset = val_dataset.select(range(min(len(val_dataset), args.eval_dataset_size)))
    input_dim = len(val_dataset[0]['expr'])
    pca_dim = len(val_dataset[0]['expr'])

    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # trust_remote_code executes custom code uploaded to the HF hub
    logger.info(f"loading model {args.model_name}")
    device = torch.device("cuda")
    if args.checkpoint is None:
        config = GPTNeoXConfig(
            hidden_size=args.hdim_2d,
            intermediate_size=args.idim_2d,
            num_attention_heads=args.nheads_2d,
            num_hidden_layers=args.nblocks_2d,
            vocab_size=100,
            use_flash_attention_2=args.use_flash_attention_2
            )
        model = GPTNeoXForCausalLM(config).to(device)

        logger.info(f"Cell encoder output dimension: {model.config.hidden_size*args.space_dim}")
        model.cell_enc = TwoLayerMLP(input_dim, model.config.hidden_size*args.space_dim)
        if args.use_vae:
            model.cell_dec = CustomVAEDecoder(
                hidden_size=model.config.hidden_size,
                input_dim=input_dim,
                device=model.device,
                reshape_postvae=args.reshape_postvae,
                space_dim=args.space_dim,
                num_blocks=1,
                mlp_enc=args.mlp_musig
            )
        else:
            model.cell_dec = nn.Linear(model.config.hidden_size*args.space_dim, input_dim)

        if args.use_pretrained:
            pretrained_model_name = "EleutherAI/pythia-160m"
            pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
            copy_weights(pretrained_model, model, config.num_hidden_layers, config.num_attention_heads, config.hidden_size, config.intermediate_size)
    else:
        config_path = os.path.join(args.checkpoint, "config.json")
        config = GPTNeoXConfig.from_pretrained(config_path)
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
        model_weights_path = os.path.join(args.checkpoint, "model.safetensors")
        pt_state_dict = safetensors.torch.load_file(model_weights_path, device="cuda")
        logger.info(model.load_state_dict(pt_state_dict))

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing. Setting use_cache to False.")
        model.config.use_cache = False

    # Get current time and initialize wandb
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"traincustom{args.train_custom}-vae{args.use_vae}-klw{args.kl_weight}-{args.model_name}-idfm{args.idfm}-hdim_2d{args.hdim_2d}idim_2d{args.idim_2d}nheads_2d{args.nheads_2d}nblocks_2d{args.nblocks_2d}-space{args.space_dim}-postvae{args.reshape_postvae}-mlpenc{args.mlp_enc}-preweights{args.use_pretrained}-pca{pca_dim}-datasize{args.train_dataset_size}-timepoints{args.time_points}-straightpath{args.straight_paths}-drop{args.dropout_p}{args.wandb_run_base_name}-{now}"

    # configure wandb logging
    if args.wandb_logging and LOCAL_RANK == 0:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
        )
        wandb.watch(model, log="all", log_freq=10)

    def data_collator(examples):
        prefixes = [examples[i]['prefix'] for i in range(len(examples))]
        paths = [examples[i]['path'] for i in range(len(examples))]

        return {
            "prefixes": prefixes,
            "paths": paths
        }
    
    def data_collator_gaussian(examples):
        exprs = [examples[i]['expr'] for i in range(len(examples))]

        return {
            "exprs": exprs,
        }


    output_dir = args.output_dir + f"/{run_name}"

    train_args = TrainingArguments(
        debug="underflow_overflow",
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        report_to="wandb",
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
    )
    logger.info(model)

    dc = data_collator
    if args.train_gaussian:
        dc = data_collator_gaussian
    
    trainer = LLMVAETrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=dc,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_context_length=args.max_context_length,
        max_num_blocks=args.max_num_blocks,
        normalize_output=args.normalize_output,
        e2e=args.e2e,
        train_gaussian=args.train_gaussian,
        time_points=args.time_points,
        use_vae=args.use_vae,
        straight_paths=args.straight_paths,
        target_dist=args.target_dist,
        target_dim=input_dim,
        just_llm=args.just_llm,
        train_custom=args.train_custom,
        ifm_reg=args.ifm_reg,
        kl_weight=args.kl_weight,
        sigma_decay=args.sigma_decay,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        dropout_p=args.dropout_p,
        scvi_dec=args.scvi_dec,
        time_scale=args.time_scale,
        scale_last=args.scale_last,
        scale_last_weight=args.scale_last_weight,
        space_dim=args.space_dim,
        reshape_postvae=args.reshape_postvae,
        idfm=args.idfm
    )

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        resume_from_checkpoint = args.checkpoint

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # only if performing distributed training over multiple instances
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)

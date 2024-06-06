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
from utils.modules import CustomDecoder, CustomVAEDecoder, CustomSCVIDecoder

from torch.nn.parallel import DistributedDataParallel as DDP

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


def main(args):

    logger.info(f"\nLOCAL RANK: {LOCAL_RANK}")
    logger.info(f"CUDA HOME: {cpp_extension.CUDA_HOME}")
    logger.info(f"\nTORCH CUDA VERSION: {torch.version.cuda}")

    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda")
    if not args.train_gaussian:
        logger.info("Loading adata...")
        adata = sc.read_h5ad(args.adata_file_path)
        logger.info(adata)

        test_combos = combo_split_nochron()
        adata.obs['cell_type_perturbation'] = list(zip(adata.obs['cell_type'], adata.obs['perturbation']))
        train_adata = adata[~adata.obs['cell_type_perturbation'].isin(test_combos)].copy()
        train_adata.obs.drop(columns=['cell_type_perturbation'], inplace=True)

        logger.info("Setting up VAE...")
        scvi.model.SCVI.setup_anndata(train_adata)
        scvi_model_cp_dir = "/".join(args.scvi_checkpoint.split("/")[:-1])
        scvi_model_cp_prefix = args.scvi_checkpoint.split("/")[-1][:-8]
        vae = scvi.model.SCVI.load(scvi_model_cp_dir, adata=train_adata, accelerator='gpu', prefix=scvi_model_cp_prefix)
        logger.info(vae)
        logger.info(f"VAE DEVICE: {vae.device}")

        with open(args.prompt_path, "r") as f:
            prompts = json.load(f)
        # scvae.train(
        #     max_epochs=10,
        #     accelerator='gpu'
        # )
        # logger.info(scvae.module)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # trust_remote_code executes custom code uploaded to the HF hub
    logger.info(f"loading model {args.model_name}")
    device = torch.device("cuda")
    if args.train_custom:
        config = GPTNeoXConfig(
            hidden_size=args.hdim_2d,
            intermediate_size=args.idim_2d,
            num_attention_heads=args.nheads_2d,
            num_hidden_layers=args.nblocks_2d,
            vocab_size=100,
            use_flash_attention_2=args.use_flash_attention_2
            )
        model = GPTNeoXForCausalLM(config).to(device)
    else:
        if not args.checkpoint:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                use_flash_attention_2=args.use_flash_attention_2).to(device)
        else:
            logger.info(f"Reloading model from checkpoint: {args.checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint,
                use_flash_attention_2=args.use_flash_attention_2).to(device)
        
        if isinstance(model, DDP):
            model = model.module

        if (args.max_context_length is not None) and (model.config.max_position_embeddings < args.max_context_length):
            logger.info(f"Max position embeddings increased.")
            model.config.max_position_embeddings = args.max_context_length
            # Need to reinstantiate rotary embeddings to adjust for change in context length
            if ("pythia" in args.model_name) and (args.checkpoint is None):

                logger.info(f"No checkpoint found, but max position embeddings increased. Reinstantiating rotary positional embeddings.")
                from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding

                for i in range(len(model.gpt_neox.layers)):
                    dim = model.gpt_neox.layers[i].attention.rotary_emb.dim
                    base = model.gpt_neox.layers[i].attention.rotary_emb.base

                    model.gpt_neox.layers[i].attention.rotary_emb = GPTNeoXRotaryEmbedding(
                        dim=dim,
                        max_position_embeddings=model.config.max_position_embeddings,
                        base=base
                    ).to(device)

    if args.gradient_checkpointing:
        logger.info(f"Using gradient checkpointing. Setting use_cache to False.")
        model.config.use_cache = False

    dataset = load_from_disk(args.llm_dataset_path)
    if args.train_dataset_size:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(args.train_dataset_size))

    if args.train_gaussian:
        split_dataset = dataset.train_test_split(
            test_size=0.1, 
            shuffle=True, 
            seed=42
        )
    else:
        filtered_dataset = dataset.filter(lambda example: (
                example['cell_type'],
                example['perturbation']
                ) not in test_combos)

        split_dataset = filtered_dataset.train_test_split(
            test_size=0.1, 
            shuffle=True, 
            seed=42
        )

    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    val_dataset = val_dataset.select(range(min(len(val_dataset), args.eval_dataset_size)))

    def preprocess_function(examples):
        # Column names 'pert_expr', 'ctr_expr', 'perturbation', 'cell_type'

        batch_size = len(examples['perturbation'])
        prefixes = []
        examples['path'] = []
        for i in range(batch_size):
            cell_type = examples['cell_type'][i]
            perturbation = examples['perturbation'][i]
            prefix = prompts['prefix'][0].format(cell_type=cell_type, perturbation=perturbation)
            ctr_expr = examples['ctr_expr'][i]
            pert_expr = examples['pert_expr'][i]
            path = generate_paths(ctr_expr, pert_expr, sigma=args.path_var, time_points=args.time_points, straight_paths=args.straight_paths)
            prefixes.append(prefix)
            examples['path'].append(path)
        examples['prefix'] = tokenizer(prefixes, truncation=True, max_length=args.max_context_length)['input_ids']
        return examples
    
    if not args.train_gaussian:
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        val_dataset = val_dataset.map(preprocess_function, batched=True)

    # Get current time and initialize wandb
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"traincustom{args.train_custom}-vae{args.use_vae}-klw{args.kl_weight}-{args.model_name}-timepoints{args.time_points}-straightpath{args.straight_paths}-drop{args.dropout_p}{args.wandb_run_base_name}-{now}"

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
        # overwrite_output_dir=args.overwrite_output_dir,
        # seed=args.seed,
        # data_seed=args.data_seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        report_to="wandb",
        # torch_compile=args.torch_compile,
        # torchdynamo=args.torchdynamo,
        # torch_compile_backend=args.torch_compile_backend,
        fp16=args.fp16,
        # ddp_backend=args.ddp_backend,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # optim=args.optim,
        # deepspeed=args.deepspeed,
        # learning_rate=args.learning_rate,
        # weight_decay=args.weight_decay
        max_steps=args.max_steps,
    )

    logger.info(f"Train Gaussian {True}")
    if args.train_gaussian:
        input_dim = len(val_dataset[0]['expr'])
        if args.train_custom:
            # input_dim = 2
            logger.info(f"Cell encoder output dimension: {model.config.hidden_size*args.space_dim}")
            model.cell_enc = nn.Linear(input_dim, model.config.hidden_size*args.space_dim)
            if args.use_vae:
                model.cell_dec = CustomVAEDecoder(
                    hidden_size=model.config.hidden_size*args.space_dim,
                    input_dim=input_dim,
                    device=model.device,
                    num_blocks=1
                )
            else:
                model.cell_dec = nn.Linear(model.config.hidden_size*args.space_dim, input_dim)
        elif args.scvi_dec:
            model.cell_enc = nn.Linear(input_dim, model.config.hidden_size).to(device)
            model.cell_dec = CustomSCVIDecoder(
                model.config.hidden_size,
                input_dim,
                device,
                num_blocks=1
            ).to(device)
        else:
            model.cell_enc = nn.Linear(input_dim, model.config.hidden_size).to(device)
            if args.use_vae:
                # model.mean_encoder = nn.Linear(model.config.hidden_size, model.config.hidden_size).to(device)
                # model.var_encoder = nn.Linear(model.config.hidden_size, model.config.hidden_size).to(device)
                # model.var_activation = vae.module.z_encoder.var_activation
                # model.var_eps = vae.module.z_encoder.var_eps
                # model.decoder = CustomDecoder(model.config.hidden_size, input_dim, device).to(device)
                model.cell_dec = CustomVAEDecoder(
                    hidden_size=model.config.hidden_size,
                    input_dim=input_dim,
                    device=model.device,
                    num_blocks=1
                )
            else:
                model.cell_dec = CustomDecoder(model.config.hidden_size, input_dim, device).to(device)
    else:
        model.cell_enc = vae.module.z_encoder.encoder.to(device)
        model.cell_proj_in = nn.Linear(model.cell_enc.fc_layers[-1][0].out_features, model.config.hidden_size).to(device)
        model.cell_proj_out = nn.Linear(model.config.hidden_size, model.cell_enc.fc_layers[-1][0].out_features).to(device)
        if not args.e2e:
            for param in model.cell_enc.parameters():
                param.requires_grad = False
        if args.normalize_output:
            model.output_norm = nn.LayerNorm(model.cell_enc.fc_layers[-1][0].out_features)
            model.output_relu = nn.ReLU()
        if args.e2e:
            model.mean_encoder = vae.module.z_encoder.mean_encoder.to(device)
            model.var_encoder = vae.module.z_encoder.var_encoder.to(device)
            model.distribution = vae.module.z_encoder.distribution
            model.var_eps = vae.module.z_encoder.var_eps
            model.return_dist = vae.module.z_encoder.return_dist
            model.z_transformation = vae.module.z_encoder.z_transformation
            model.var_activation = vae.module.z_encoder.var_activation
            model.px_r = vae.module.px_r
            model.decoder = vae.module.decoder.to(device)

    logger.info(model)
    # logger.info(torch.cuda.device_count())
    # if torch.cuda.device_count() > 1:
    #     accelerator = Accelerator()
    #     model = accelerator.prepare(model)

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
        space_dim=args.space_dim
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

import argparse
import json
import logging
import os
import pickle
import random
from datetime import datetime
from tqdm import tqdm

import anndata
import scanpy as sc
import scvi
import torch
import wandb
from datasets import load_from_disk
from torch import nn
from torch.distributions import Normal
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
from utils.custom_deterministic_trainer import DeterministicLLMVAETrainer
from utils.modules import CustomDecoder

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
        default="/home/dfl32/scratch/cinemaot_data/ifm_hf_ds/hf_one_ep_ds"
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
        "--straight_paths",
        type=bool,
        default=False
    )
    return parser.parse_args()

def main(args):

    logger.info(f"\nLOCAL RANK: {LOCAL_RANK}")
    logger.info(f"CUDA HOME: {cpp_extension.CUDA_HOME}")
    logger.info(f"\nTORCH CUDA VERSION: {torch.version.cuda}")

    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda")

    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)

    # trust_remote_code executes custom code uploaded to the HF hub
    logger.info(f"loading model {args.model_name}")
    device = torch.device("cuda")
    if not args.checkpoint:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            use_flash_attention_2=args.use_flash_attention_2).to(device)
    else:
        logger.info(f"Reloading model from checkpoint: {args.checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            use_flash_attention_2=args.use_flash_attention_2).to(device)

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
        dataset = dataset.select(range(args.train_dataset_size))

    split_dataset = dataset.train_test_split(
        test_size=0.1, 
        shuffle=True, 
        seed=42
    )

    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    val_dataset = val_dataset.select(range(min(len(val_dataset), args.eval_dataset_size)))

    # Get current time and initialize wandb
    now = datetime.now()
    now = datetime.strftime(now, "%Y-%m-%d_%H-%M-%S")
    run_name = f"cinemaot-{args.model_name}-timepoints{args.time_points}-{args.wandb_run_base_name}-{now}"

    # configure wandb logging
    if args.wandb_logging and LOCAL_RANK == 0:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
        )
        wandb.watch(model, log="all", log_freq=10)
    
    def data_collator(examples):
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
    )

    # Add encoder/decoder for model
    input_dim = len(val_dataset[0]['expr'])
    model.cell_enc = nn.Linear(input_dim, model.config.hidden_size).to(device)
    model.cell_dec = CustomDecoder(model.config.hidden_size, input_dim, device).to(device)

    logger.info(model)
    
    trainer = DeterministicLLMVAETrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        time_points=args.time_points,
        straight_paths=args.straight_paths
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

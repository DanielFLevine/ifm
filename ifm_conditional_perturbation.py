import json
import logging
import os
from datetime import datetime

import safetensors
import torch
import wandb
from datasets import load_from_disk
from torch.utils import cpp_extension
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    GPTNeoXForCausalLM,
    GPTNeoXConfig
)

from utils.conditional_custom_trainer import ConditionalLLMVAETrainer
from utils.modules import CustomVAEDecoder, TwoLayerMLP
from utils.parser import parse_arguments
from ipdb import set_trace


logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/home/dfl32/project/huggingface"
if "LOCAL_RANK" in os.environ:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
else:
    LOCAL_RANK = 0


def main(args):

    logger.info(f"\nLOCAL RANK: {LOCAL_RANK}")
    logger.info(f"CUDA HOME: {cpp_extension.CUDA_HOME}")
    logger.info(f"\nTORCH CUDA VERSION: {torch.version.cuda}")

    with open(args.data_paths, 'r') as f:
        data_paths = json.load(f)
    
    dataset_path = data_paths[args.pert_split]
    train_dataset = load_from_disk(os.path.join(dataset_path, 'train_ds'))
    val_dataset = load_from_disk(os.path.join(dataset_path, 'val_ds'))
    val_dataset = val_dataset.select(range(min(len(val_dataset), args.eval_dataset_size)))
    input_dim = len(val_dataset[0]['expression'])
    # set_trace()
    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # trust_remote_code executes custom code uploaded to the HF hub
    logger.info(f"loading model {args.model_name}")
    device = torch.device("cuda")
    if args.checkpoint is None:
        model = GPTNeoXForCausalLM.from_pretrained(
            args.model_name,
            use_flash_attention_2=args.use_flash_attention_2
            ).to(device)

        logger.info(f"Cell encoder output dimension: {model.config.hidden_size*args.space_dim}")
        model.cell_enc = TwoLayerMLP(input_dim, model.config.hidden_size*args.space_dim)
        model.cell_dec = CustomVAEDecoder(
            hidden_size=model.config.hidden_size,
            input_dim=input_dim,
            device=model.device,
            reshape_postvae=True,
            space_dim=args.space_dim,
            num_blocks=1,
            mlp_enc=True
        )
    else:
        config_path = os.path.join(args.checkpoint, "config.json")
        config = GPTNeoXConfig.from_pretrained(config_path)
        model = GPTNeoXForCausalLM(config).to(device)
        model.cell_enc = TwoLayerMLP(input_dim, model.config.hidden_size*args.space_dim).to(device)
        model.cell_dec = CustomVAEDecoder(
            hidden_size=config.hidden_size,
            input_dim=input_dim,
            device=device,
            reshape_postvae=True,
            space_dim=args.space_dim,
            num_blocks=1,
            mlp_enc=True
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
    run_name = f"conditional_gen-klw{args.kl_weight}-{args.model_name}-space{args.space_dim}-pps{args.points_per_sample}-split{args.pert_split}-{args.wandb_run_base_name}-{now}"

    if args.wandb_logging:
        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
        )
        wandb.watch(model, log="all", log_freq=10)


    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)

    def data_collator(examples):
        # Columns 'expression', 'cell_type', 'perturbation', 'chronicity'
        expressions = [examples[i]['expression'] for i in range(len(examples))]
        cell_types = [examples[i]['cell_type'] for i in range(len(examples))]
        perturbations = [examples[i]['perturbation'] for i in range(len(examples))]
        chronicities = [examples[i]['chronicity'] for i in range(len(examples))]

        return {
            "expression": expressions,
            "cell_type": cell_types,
            "perturbation": perturbations,
            "chronicity": chronicities
        }

    output_dir = args.output_dir + f"/{run_name}"
    output_dir = os.path.join(output_dir, args.pert_split)
    os.makedirs(output_dir, exist_ok=True)

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
    
    trainer = ConditionalLLMVAETrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=dc,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_context_length=args.max_context_length,
        max_num_blocks=args.max_num_blocks,
        time_points=args.time_points,
        straight_paths=args.straight_paths,
        kl_weight=args.kl_weight,
        sigma_decay=args.sigma_decay,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        space_dim=args.space_dim,
        prompt_path=args.prompt_path,
        samples_per_input=args.samples_per_input
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

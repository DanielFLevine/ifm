import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_paths",
        type=str,
        default="/home/dfl32/project/ifm/cinemaot_data/data_paths.json",
        help="Path to json file containing paths to datasets"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="/home/dfl32/project/ifm/prompts/cinemaot_prompts.json",
        help="Path to json file containing prompts"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/dfl32/scratch/training-runs",
        help="Where to output saved checkpoints"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-160m",
        help="Hugging Face model name for pretrained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action='store_true',
        help="Whether to resume training from checkpoint or start training from scratch"
    )
    parser.add_argument(
        "--time_points",
        type=int,
        default=16,
        help="Number of time points in each path"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=50,
        help="Maximum context length of LLM"
    )
    parser.add_argument(
        "--max_num_blocks",
        type=int,
        default=32,
        help="Maximum number of blocks per batch during training"
    )
    parser.add_argument(
        "--use_flash_attention_2",
        action='store_true',
        help="Whether to use flash attention 2 for training. Recommended True"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action='store_true',
        help="Whether to use gradient checkpointing for training. Recommended True"
    )
    parser.add_argument(
        "--eval_dataset_size",
        type=int,
        default=1000,
        help="Limit validation dataset size to speed up evaluation"
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="IFM",
        help="Project name used in wandb"
    )
    parser.add_argument(
        "--wandb_logging",
        action='store_true',
        help="Whether to log to wandb."
    )
    parser.add_argument(
        "--wandb_run_base_name",
        type=str,
        default="ifm",
        help="Base name for wandb run"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=10,
        help="How many samples to put on each GPU for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=10,
        help="How many samples to put on each GPU for evaluation"
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="What measure to use to determine when to evaluate. See Hugging Face for more details"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="How many steps between each evaluation. See Hugging Face for more details"
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        type=int,
        default=5,
        help="Number of forward passes before computing evaluation metrics. See Hugging Face for more details"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="Whether to use float16 for training. Recommended True"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of workings for moving between CPU and GPU memory"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of forward passes before computing loss"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of steps before logging training metrics"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        help="What measure to use to determine when to save a model checkpoint. See Hugging Face for more details"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between checkpoint saves"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=100,
        help="Maximum number of saved checkpoints allowed in output directory"
    )
    parser.add_argument(
        "--straight_paths",
        action='store_true',
        help="Whether to use straight paths with no noise during training. Recommended True"
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="Weighting for KL divergence in loss term"
    )
    parser.add_argument(
        "--sigma_decay",
        action='store_true',
        help="Whether to decay path noise when generating noisy paths"
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.0001,
        help="Minimum STD when decaying path noise"
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=0.01,
        help="Maximum STD when decaying path noise"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps to perform. WARNING: Overrides num_train_epochs"
    )
    parser.add_argument(
        "--space_dim",
        type=int,
        default=1,
        help="Number of tokens that represent a single time step"
    )
    parser.add_argument(
        "--pert_split",
        type=str,
        default="ct_pert",
        help="Which train/test split to use for conditional generation. Valid values are 'ct_pert' or 'chron'"
    )
    parser.add_argument(
        "--samples_per_input",
        type=int,
        default=1,
        help="Number of samples per single input into LLM. Reshapes batch by diving by this number by the current batch size"
    )
    parser.add_argument(
        "--points_per_sample",
        type=int,
        default=1,
        help="Number of flows per sample. Making this > 1 turns on multipoint integration"
    )
    return parser.parse_args()
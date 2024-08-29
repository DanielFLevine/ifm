#!/bin/bash
#SBATCH --job-name=ifm_cond               # Job name
#SBATCH --output logs/ifm_cond_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
#SBATCH --partition gpu_devel
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb                                 # Job memory request
#SBATCH --time=6:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export TOKENIZERS_PARALLELISM=true
export SPACE=1
export TIME_POINTS=16
export POINTS_PER_SAMPLE=1
export SAMPLES_PER_SEQUENCE=2
export PER_DEVICE_BATCH_SIZE=128

python ifm_conditional_perturbation.py \
    --data_paths /home/dfl32/project/ifm/cinemaot_data/data_paths.json \
    --prompt_path /home/dfl32/project/ifm/prompts/cinemaot_prompts.json \
    --model_name EleutherAI/pythia-160m \
    --num_train_epochs 100 \
    --time_points $TIME_POINTS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --eval_accumulation_steps 5 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --gradient_checkpointing \
    --wandb_logging \
    --use_flash_attention_2 \
    --fp16 \
    --straight_paths \
    --kl_weight 0.3 \
    --space_dim $SPACE \
    --points_per_sample $POINTS_PER_SAMPLE
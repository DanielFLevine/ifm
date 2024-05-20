#!/bin/bash
#SBATCH --job-name=ifm_pert               # Job name
#SBATCH --output logs/ifm_pert_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
#SBATCH --partition gpu
#SBATCH --reservation=dijk
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb                                 # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --nnodes=1 ifm_perturbation.py \
    --model_name EleutherAI/pythia-160m \
    --llm_dataset_path /home/dfl32/scratch/cinemaot_data/ifm_hf_ds/gaussian_768_hf_ds \
    --num_train_epochs 100 \
    --train_gaussian True \
    --time_points 32 \
    --max_context_length 50 \
    --max_num_blocks 32 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --eval_accumulation_steps 5 \
    --gradient_accumulation_steps 1 \
    --normalize_output True \
    --save_steps 500 \
    --e2e True \
    --straight_paths True \
    --use_vae True \
    --kl_weight 0.3 \
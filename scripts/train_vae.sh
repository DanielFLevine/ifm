#!/bin/bash
#SBATCH --job-name=vae              # Job name
#SBATCH --output logs/vae_%J.log    # Output log file
#SBATCH --mail-type=ALL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu  # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1         # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb                  # Job memory request
#SBATCH --time=2-00:00:00           # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export TOKENIZERS_PARALLELISM=true

python train_vae.py \
    --llm_dataset_path /home/dfl32/scratch/cinemaot_data/ifm_hf_ds/gaussian_pca1000_hf_ds_new \
    --input_dim 1000 \
    --hidden_dim 4096 \
    --latent_dim 256 \
    --num_train_steps 1000000 \
    --batch_size 128 \
    --wandb_log_steps 5000 \
    --eval_dataset_size 1000 \
    --beta 0.3
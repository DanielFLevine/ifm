#!/bin/bash
#SBATCH --job-name=corrs_vae             # Job name
#SBATCH --output logs/corrs_vae_%J.log   # Output log file
#SBATCH --mail-type=ALL                  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu  # Where to send mail
#SBATCH --partition gpu_devel
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1              # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb                       # Job memory request
#SBATCH --time=6:00:00                   # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export TOKENIZERS_PARALLELISM=true
export CP_DIR=/home/dfl32/scratch/training-runs/vae/vae-1000-4096-128
export CHECKPOINT=1000000

python compute_unconditional_corrs_vae.py \
    --cp_dir $CP_DIR \
    --checkpoint $CHECKPOINT \
    --num_samples 20000 \
    --num_repeats 5 \
    --input_dim 1000 \
    --hidden_dim 4096 \
    --latent_dim 128 \
    --hvgs 50 \
    --num_pca_dims 10 \
    --umap_embed \
    --plot_umap \
    --mmd_gamma 1.0 \
    --wass_reg 0.01 \
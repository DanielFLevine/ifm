#!/bin/bash
#SBATCH --job-name=mmd_wass_vae             # Job name
#SBATCH --output logs/mmd_wass_vae_%J.log   # Output log file
#SBATCH --mail-type=ALL                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu  # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                 # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb                         # Job memory request
#SBATCH --time=2-00:00:00                   # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export TOKENIZERS_PARALLELISM=true
export CP_DIR=/home/dfl32/scratch/training-runs/vae/vae-1000-4096-128
export CHECKPOINT=1000000

# Define the parameters
export TOTAL_SAMPLES=20000
mmd_gammas="0.01 0.1 1.0 2.0 10.0"
wass_regs="0.001 0.01 0.1 1.0 10.0"

python compute_mmd_wass_vae.py \
    --cp_dir $CP_DIR \
    --checkpoint $CHECKPOINT \
    --num_samples $TOTAL_SAMPLES \
    --mmd_gammas $mmd_gammas \
    --wass_regs $wass_regs \
    --umap_embed \
    --num_umap_dims 5 \
    --input_dim 1000 \
    --hidden_dim 4096 \
    --latent_dim 128 \
    --batch_size 100 \
    --num_repeats 5
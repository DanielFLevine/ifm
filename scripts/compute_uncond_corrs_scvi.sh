#!/bin/bash
#SBATCH --job-name=corrs             # Job name
#SBATCH --output logs/corrs_scvi_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb                                 # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm

python compute_unconditional_corrs_scvi.py \
    --num_samples 20000 \
    --num_repeats 1 \
    --hvgs 50 \
    --n_cell_thresh 100 \
    --num_pca_dims 10 \
    --umap_embed \
    --plot_umap \
    --mmd_gamma 1.0 \
    --wass_reg 0.01 \
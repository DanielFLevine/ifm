#!/bin/bash
#SBATCH --job-name=scvi              # Job name
#SBATCH --output logs/scvi_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1                   
#SBATCH --cpus-per-task=2
#SBATCH --mem=100gb                               # Job memory request
#SBATCH --time=1-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
conda activate c2s2
cd /home/dfl32/project/ifm

python train_scvi.py \
    --adata_path /home/dfl32/project/ifm/cinemaot_data/ifm_adatas/train.h5ad \
    --num_epochs 100 \
    --n_layers 2 \
    --n_hidden 128 \
    --n_latent 10 \
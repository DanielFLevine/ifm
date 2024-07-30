#!/bin/bash
#SBATCH --job-name=corrs             # Job name
#SBATCH --output logs/corrs_diff_%J.log        # Output log file
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
export TOKENIZERS_PARALLELISM=true
export CP_DIR=/home/dfl32/scratch/training-runs/simple_ifm/diffusion-inputdim1000-2024-07-09_20-47-39
export CHECKPOINT=2000000

python compute_unconditional_corrs_diff.py \
    --cp_dir $CP_DIR \
    --checkpoint $CHECKPOINT \
    --num_samples 20000 \
    --num_repeats 5 \
    --input_dim 1000 \
    --hvgs 50 \

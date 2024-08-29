#!/bin/bash
#SBATCH --job-name=cfm              # Job name
#SBATCH --output logs/moons_cfm_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb                                 # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm

python train_cfm_2d.py \
    --batch_size 256 \
    --inf_batch_size 1024 \
    --path_sigma 0.5 \
    --lr 0.01 \
    --timepoints 100 \
    --ode_solver euler \
    --type sb
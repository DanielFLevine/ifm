#!/bin/bash
#SBATCH --job-name=idfm              # Job name
#SBATCH --output logs/moons_%J.log        # Output log file
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

python idfm_multipoint.py \
    --batch_size 256 \
    --inf_batch_size 256 \
    --timepoints 100 \
    --num_steps 100000 \
    --hdim 64 \
    --nlayer 2 \
    --nhead 2 \
    --lr 0.0001 \
    --continuous_time \
    --ve \
    --attn_dropout 0.95 \
    --sigma_max 10.0 \
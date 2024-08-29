#!/bin/bash
#SBATCH --job-name=2d_calmflow             # Job name
#SBATCH --output logs/moons_%J.log        # Output log file
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

python calmflow_2d.py \
    --batch_size 2048 \
    --inf_batch_size 256 \
    --timepoints 16 \
    --num_steps 100000 \
    --hdim 256 \
    --nlayer 2 \
    --nhead 4 \
    --lr 0.0001 \
    --idfm \
    --random_labels \
    --plot_train_trajs \
    --path_sigma 0.1 \
    --predict_solver_step
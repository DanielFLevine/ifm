#!/bin/bash
#SBATCH --job-name=mlp_training          # Job name
#SBATCH --output=logs/mlp_training_%J.log # Output log file
#SBATCH --mail-type=ALL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu # Where to send mail
#SBATCH --partition=gpu                   # Partition to use
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks per node
#SBATCH --gpus=1                          # Number of GPUs
#SBATCH --cpus-per-task=2                 # Number of CPU cores per task
#SBATCH --mem=64gb                        # Memory per node
#SBATCH --time=6:00:00                    # Time limit hrs:min:sec
date;hostname;pwd

# Load necessary modules
module load miniconda
module load CUDA/12.1  # Load CUDA 12.1

# Activate conda environment
source activate c2s2  # Activate the conda environment c2s2

# Change to the directory containing your script
cd /home/dfl32/project/ifm

# Run the Python script with arguments
python train_classifier.py \
    --num_steps 100000 \
    --batch_size 64 \
    --hidden_dim 256 \
    --learning_rate 0.001 \
    --steps_per_checkpoint 1000 \
    --leiden 1
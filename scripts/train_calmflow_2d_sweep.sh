#!/bin/bash
#SBATCH --job-name=2d_calmflow             # Job name
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
#SBATCH --time=6:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
echo CUDA_LAUNCH BLOCKING is $CUDA_LAUNCH_BLOCKING
wandb login d3ca39e50076ef0bbe4585d85f4c66de3a4adf35
export SWEEP_ID=$(wandb sweep calmflow_2d_sweep.yaml 2>&1 | awk '/with:/{print $8}')
echo SWEEP_ID is $SWEEP_ID
wandb agent $SWEEP_ID
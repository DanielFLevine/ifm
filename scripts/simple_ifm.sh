#!/bin/bash
#SBATCH --job-name=simple_ifm               # Job name
#SBATCH --output logs/simple_ifm_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb                                 # Job memory request
#SBATCH --time=6:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export TOKENIZERS_PARALLELISM=true

python simple_ifm.py \
#!/bin/bash
#SBATCH --job-name=diffusion              # Job name
#SBATCH --output logs/diffusion_%J.log        # Output log file
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
export TOKENIZERS_PARALLELISM=true

python train_diffusion.py \
    --llm_dataset_path /home/dfl32/scratch/cinemaot_data/ifm_hf_ds/gaussian_pca768_normFalse_scaleFalse_minmaxTrue_hf_ds \
    --denoising_time_steps 1000 \
    --batch_size 512\
    --num_train_steps 1000000 \
    --save_steps 5000 \
    --intermediate_dim 2048 \
    --num_fc_layers 2
    --input_dim 1000
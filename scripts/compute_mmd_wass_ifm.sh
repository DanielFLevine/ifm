#!/bin/bash
#SBATCH --job-name=mmd_wass             # Job name
#SBATCH --output logs/mmd_wass_%J.log   # Output log file
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu  # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1             # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
module load CUDA/12.1
conda activate c2s2
cd /home/dfl32/project/ifm
export TOKENIZERS_PARALLELISM=true

# Define the parameters
export TOTAL_SAMPLES=20000
mmd_gammas="0.01 0.1 1.0 2.0 10.0"
wass_regs="0.001 0.01 0.1 1.0 10.0"
space_dim=1

python compute_mmd_wass_ifm.py \
    --model_json_path /home/dfl32/project/ifm/models/ifm_paths2.json \
    --num_samples $TOTAL_SAMPLES \
    --mmd_gammas $mmd_gammas \
    --wass_regs $wass_regs \
    --space_dim $space_dim \
    --mlp_enc \
    --mlp_musig \
    --reshape_postvae \
    --temp 1.7 \
    --umap_embed \
    --num_umap_dims 10 \
    --pretrained_weights \
    --num_repeats 5 \
    --leiden 1
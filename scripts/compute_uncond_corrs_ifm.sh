#!/bin/bash
#SBATCH --job-name=corrs             # Job name
#SBATCH --output logs/corrs_ifm_%J.log        # Output log file
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
export SPACE_DIM=10
export TOTAL_SAMPLES=20000
export POINTS_PER_SAMPLE=1

python compute_unconditional_corrs_ifm.py \
    --model_json_path /home/dfl32/project/ifm/models/ifm_big_models.json \
    --num_samples $(($TOTAL_SAMPLES / $POINTS_PER_SAMPLE)) \
    --input_dim 1000 \
    --temp 1.0 \
    --batch_size 100 \
    --num_repeats 5 \
    --hvgs 50 \
    --time_points 16 \
    --space_dim $SPACE_DIM \
    --reshape_postvae \
    --mlp_enc \
    --mlp_musig \
    --mmd_gamma 1.0 \
    --num_pca_dims 10 \
    --umap_embed \
    --points_per_sample $POINTS_PER_SAMPLE
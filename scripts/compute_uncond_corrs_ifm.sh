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
export CP_DIR=/home/dfl32/scratch/training-runs/traincustomTrue-vaeTrue-klw0.3-EleutherAI
export DATE=2024-07-11_10-33-12
export SPACE_DIM=20
export CHECKPOINT=pythia-160m-idfmFalse-hdim_2d64idim_2d64nheads_2d4nblocks_2d2-space${SPACE_DIM}-postvaeTrue-mlpencTrue-preweightsTrue-pca1000-datasizeNone-timepoints16-straightpathTrue-drop0.0ifm-${DATE}

python compute_unconditional_corrs_ifm.py \
    --cp_dir $CP_DIR \
    --checkpoint $CHECKPOINT \
    --num_samples 20000 \
    --input_dim 1000 \
    --temp 0.5 \
    --batch_size 100 \
    --num_repeats 5 \
    --hvgs 50 \
    --time_points 128 \
    --space_dim $SPACE_DIM \
    --reshape_postvae True \
    --mlp_enc True \
    --mlp_musig True \
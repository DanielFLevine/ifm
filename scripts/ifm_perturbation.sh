#!/bin/bash
#SBATCH --job-name=ifm_pert               # Job name
#SBATCH --output logs/ifm_pert_%J.log        # Output log file
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
export SPACE=1
export DT=2024-07-09_17-10-16
export CP_DIR=/home/dfl32/scratch/training-runs/traincustomTrue-vaeTrue-klw0.3-EleutherAI
export CP_MODEL=pythia-160m-idfmFalse-hdim_2d64idim_2d64nheads_2d4nblocks_2d2-space${SPACE}-postvaeTrue-mlpencTrue-preweightsTrue-pca1000-datasizeNone-timepoints16-straightpathTrue-drop0.0ifm-${DT}

python ifm_perturbation_clean.py \
    --model_name EleutherAI/pythia-160m \
    --llm_dataset_path /home/dfl32/scratch/cinemaot_data/ifm_hf_ds/gaussian_pca1000_hf_ds_new \
    --train_gaussian True \
    --num_train_epochs 200 \
    --time_points 16 \
    --max_context_length 50 \
    --max_num_blocks 32 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --eval_accumulation_steps 5 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --e2e True \
    --train_custom True \
    --hdim_2d 64 \
    --idim_2d 64 \
    --nheads_2d 4 \
    --nblocks_2d 2 \
    --straight_paths True \
    --use_vae True \
    --kl_weight 0.3 \
    --space_dim $SPACE \
    --reshape_postvae True \
    --mlp_enc True \
    --mlp_musig True \
    # --use_pretrained True \
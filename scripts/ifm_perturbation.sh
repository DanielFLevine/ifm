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
export SPACE=50
export TIME_POINTS=16
export POINTS_PER_SAMPLE=4
export SAMPLES_PER_SEQUENCE=2
export DT=2024-07-09_17-10-16
export PER_DEVICE_BATCH_SIZE=128
export CP_DIR=/home/dfl32/scratch/training-runs/traincustomTrue-vaeTrue-klw0.3-EleutherAI
export CP_MODEL=pythia-160m-idfmFalse-hdim_2d64idim_2d64nheads_2d4nblocks_2d2-space${SPACE}-postvaeTrue-mlpencTrue-preweightsTrue-pca1000-datasizeNone-timepoints16-straightpathTrue-drop0.0ifm-${DT}

python ifm_perturbation_clean.py \
    --model_name EleutherAI/pythia-160m \
    --llm_dataset_path /home/dfl32/scratch/cinemaot_data/ifm_hf_ds/gaussian_pca1000_hf_ds_new \
    --train_gaussian \
    --num_train_epochs 400 \
    --time_points $TIME_POINTS \
    --max_context_length $(($TIME_POINTS * $POINTS_PER_SAMPLE * $SPACE * $SAMPLES_PER_SEQUENCE)) \
    --max_num_blocks $(($PER_DEVICE_BATCH_SIZE / $SAMPLES_PER_SEQUENCE)) \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --eval_accumulation_steps 5 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --e2e \
    --train_custom \
    --hdim_2d 64 \
    --idim_2d 64 \
    --nheads_2d 4 \
    --nblocks_2d 2 \
    --straight_paths \
    --use_vae \
    --kl_weight 0.3 \
    --space_dim $SPACE \
    --reshape_postvae \
    --mlp_enc \
    --mlp_musig \
    --use_pretrained \
    --points_per_sample $POINTS_PER_SAMPLE
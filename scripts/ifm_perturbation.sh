#!/bin/bash
#SBATCH --job-name=ifm_pert               # Job name
#SBATCH --output logs/ifm_pert_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=conrad.lee@yale.edu                       # Where to send mail
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
conda activate ifm
cd /gpfs/gibbs/project/dijk/ccl66/ifm
export TOKENIZERS_PARALLELISM=true


python ifm_perturbation.py \
    --model_name EleutherAI/pythia-160m \
    --llm_dataset_path /home/dfl32/palmer_scratch/ifm/cinemaot_data/ifm_hf_ds/gaussian_768_hf_ds \
    --train_gaussian True \
    --num_train_epochs 100 \
    --time_points 16 \
    --max_context_length 50 \
    --max_num_blocks 32 \
    --train_dataset_size 5000 \
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
    --scale_last True \
    --output_dir /home/ccl66/palmer_scratch/ifm/training-runs
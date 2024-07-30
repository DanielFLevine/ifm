#!/bin/bash
#SBATCH --job-name=download             # Job name
#SBATCH --output logs/download_%J.log        # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.levine@yale.edu                       # Where to send mail
#SBATCH --partition bigmem
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb                                 # Job memory request
#SBATCH --time=1-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

cd /home/dfl32/scratch/amd_data/genomewide_perterbseq

wget https://plus.figshare.com/ndownloader/articles/20029387/versions/1
# wget https://zenodo.org/api/records/7041849/files-archive
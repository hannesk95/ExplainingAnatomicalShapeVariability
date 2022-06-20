#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-8:0:0
#SBATCH --account=def-uofavis-ab
#SBATCH --mail-user=kiechle@ualberta.ca
#SBATCH --mail-type=BEGIN,END

#load python 3.9
module load python/3.9

#activate virtual environment
cd && source projects/def-uofavis-ab/jkiechle/CoMA/env/bin/activate

#pull latest version of GIT repository
cd && cd projects/def-uofavis-ab/jkiechle/CoMA/Hippocampus_CoMA/ && git checkout dev && git pull

#execute main script
cd /src/ && python main.py
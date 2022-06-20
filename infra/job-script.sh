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

module load python/3.8

source projects/def-uofavis-ab/jkiechle/CoMA/env/bin/activate

#tar -xzf ~/projects/def-uofavis-ab/dmiller/COMA_processed.tar.gz -C $SLURM_TMPDIR

#python main.py --data_dir $SLURM_TMPDIR/COMA_data --split sliced --split_term sliced

#!/bin/bash
#SBATCH --job-name=xleap_wiggle
#SBATCH --output=logs/wiggle_%j.out
#SBATCH --error=logs/wiggle_%j.err
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition milano
#SBATCH --account facet

# Activate your psana/python virtual environment here
# source activate custom_psana_env

# Run analysis on the pickle bundle generated for run 145 across target classes
python3 run_analysis_with_slurm.py --pickle_file /sdf/scratch/users/j/jinseop/preprocessed_run_145.pkl --mask goose

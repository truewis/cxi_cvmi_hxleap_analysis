#!/bin/bash
#SBATCH --job-name=xleap_wiggle
#SBATCH --output=logs/wiggle_%j.out
#SBATCH --error=logs/wiggle_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --memory=16G

# Activate your psana/python virtual environment here
# source activate custom_psana_env

# Run analysis on the pickle bundle generated for run 145 across target classes
python3 run_analysis_with_slurm.py --pickle_file ./preprocessed_run_145.pkl --mask goose

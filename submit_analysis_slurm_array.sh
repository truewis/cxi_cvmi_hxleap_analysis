#!/bin/bash
#SBATCH --job-name=xleap_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --memory=8G
#SBATCH --array=0-19   # This triggers 6 distinct parallel jobs (Step 0 through Step 5)

# Each separate parallel node will catch its own task index number and run just that step:
python3 run_analysis_with_slurm.py --pickle_file ./preprocessed_run_145.pkl --mask goose --steps $SLURM_ARRAY_TASK_ID

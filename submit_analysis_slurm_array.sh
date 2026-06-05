#!/bin/bash
#SBATCH --job-name=xleap_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition milano
#SBATCH --account facet
#SBATCH --array=0-19   # This triggers distinct parallel jobs (Step 0 through Step 5)

# Each separate parallel node will catch its own task index number and run just that step:
python3 run_analysis_with_slurm.py --pickle_file /sdf/scratch/users/j/jinseop/preprocessed_run_145.pkl --mask duck --steps $SLURM_ARRAY_TASK_ID --r_adjustment -1

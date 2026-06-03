#!/bin/bash
#SBATCH --job-name=mock_array_sim
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --memory=8G
#SBATCH --array=0-7

# Array containing all density factors (i)
I_CHOICES=(5 8 11 14 17 20 23 26)
CURRENT_I=${I_CHOICES[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

# Call the standalone wrapper script
python3 run_bootstrap_with_slurm.py --i_values $CURRENT_I --iterations 100

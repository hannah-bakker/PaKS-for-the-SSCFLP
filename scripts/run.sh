#!/bin/bash
#SBATCH --job-name=exp_%x
#SBATCH --partition=cpu
#SBATCH --time=02:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

INST="/pfs/data6/home/ka/ka_ior/ka_ma2351/paks20/data/cflp-tbed1/i300_1.json"

# Activate virtualenv
source "$HOME/paks-sscflp/venvs/paks-cplex/bin/activate"

# Go to repo root (fail hard if it doesn't exist)
cd "$HOME/paks-sscflp/PaKS-for-the-SSCFLP" || exit 1

# Ensure log directory exists (for SLURM output/error)
mkdir -p logs


srun python scripts/main.py \
    --path_to_instance "$INST" \
    --timelimit "3600" \

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ham10000.out
#SBATCH --error=logs/ham10000.err
#SBATCH --job-name="HAM10000"

srun singularity exec --nv containers/container.sif python src/data/ham10000.py
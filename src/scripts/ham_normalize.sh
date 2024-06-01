#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ham_normalize.out
#SBATCH --error=logs/ham_normalize.err
#SBATCH --job-name="HAM NORM"

srun singularity exec --nv containers/container.sif python src/data/ham_normalization.py
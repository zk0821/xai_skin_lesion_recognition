#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/ci_net.out
#SBATCH --error=logs/ci_net.err
#SBATCH --job-name="CI-Net"

srun singularity exec --nv containers/container.sif python src/models/ci_net/ci_net.py
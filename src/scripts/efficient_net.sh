#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/efficient_net.out
#SBATCH --error=logs/efficient_net.err
#SBATCH --job-name="EfficientNet"

srun singularity exec --nv containers/container.sif python src/models/transfer_learning/efficient_net.py
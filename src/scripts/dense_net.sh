#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/dense_net.out
#SBATCH --error=logs/dense_net.err
#SBATCH --job-name="DenseNet"

srun singularity exec --nv containers/container.sif python src/models/transfer_learning/dense_net.py
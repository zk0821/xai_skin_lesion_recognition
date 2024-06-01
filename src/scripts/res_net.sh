#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/res_net.out
#SBATCH --error=logs/res_net.err
#SBATCH --job-name="ResNet"

srun singularity exec --nv containers/container.sif python src/models/transfer_learning/res_net.py
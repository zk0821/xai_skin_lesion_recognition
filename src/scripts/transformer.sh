#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/transformer.out
#SBATCH --error=logs/transformer.err
#SBATCH --job-name="Transformer"

srun singularity exec --nv containers/container.sif python src/models/transformer/swin_transformer.py
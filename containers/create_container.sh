singularity build containers/container.sif docker://pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
singularity exec containers/container.sif pip install install torch torchvision torchaudio
singularity exec containers/container.sif pip install requests
singularity exec containers/container.sif pip install opencv-python-headless
singularity exec containers/container.sif pip install matplotlib
singularity exec containers/container.sif pip install seaborn
singularity exec containers/container.sif pip install scikit-image
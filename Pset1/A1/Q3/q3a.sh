#!/bin/bash
#SBATCH --job-name=ndvi_gpu
#SBATCH --output=ndvi_gpu_%j.out
#SBATCH --error=ndvi_gpu_%j.err

#SBATCH --account=macs30123
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=00:10:00
#SBATCH --mem=8G

# Load Python and GPU-related modules
module load python
module load cuda
module load gcc
# (Optional) Activate your virtual environment if needed
# source ~/.venv/bin/activate

# Run your GPU NDVI Python script
python3 q3a.py
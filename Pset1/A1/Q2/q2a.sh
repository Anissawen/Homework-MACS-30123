#!/bin/bash
#SBATCH --job-name=q2b
#SBATCH --output=q2b.out
#SBATCH --error=q2b.err

#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

module load python
module load mpich

mpirun -n 10 python3 q2a.py

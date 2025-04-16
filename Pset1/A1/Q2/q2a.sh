#!/bin/bash
#SBATCH --job-name=q1b_variable_cores
#SBATCH --output=q1b_core_%j.out
#SBATCH --error=q1b_core_%j.err

#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

#SBATCH --mail-type=END
#SBATCH --mail-user=anissa6@rcc.uchicago.edu

module load python
module load mpich
# source activate your_env_name  # if needed

mpirun -n 10 python3 q2a.py

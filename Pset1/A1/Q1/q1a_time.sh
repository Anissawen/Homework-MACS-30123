#!/bin/bash
#SBATCH --job-name=q1a_time
#SBATCH --output=q1a_time.out
#SBATCH --error=q1a_time.err
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END
#SBATCH --mail-user=anissa6@rcc.uchicago.edu
# Load required modules
module load python

python q1a.py build_ext --inplace

# Run the simulation script
python q1a_time.py
#!/bin/bash
#SBATCH --job-name=q1b
#SBATCH --output=q1b.out
#SBATCH --error=q1b.err

#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

#SBATCH --mail-type=END
#SBATCH --mail-user=anissa6@rcc.uchicago.edu

module load python
module load mpich

for i in {1..20}
do
  echo "Running simulation with $i core(s)..."
  mpirun -n $i python3 pset1.py
done

python q1b_plot.py
#!/bin/bash
#
#SBATCH --job-name=PodPy-Test
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --threads-per-core=1

module purge
source prospero-modules.sh
source activate

python test.py

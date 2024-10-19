#!/bin/bash
#SBATCH -J hubb2d
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=12:00:00

module load slurm_setup
export OMP_NUM_THREADS=28
mpiexec -n $SLURM_NTASKS ./hubbard2d.exe

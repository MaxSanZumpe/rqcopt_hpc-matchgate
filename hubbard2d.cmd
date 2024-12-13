#!/bin/bash
#SBATCH -J hubb2d
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --
#SBATCH --export=NONE
#SBATCH --time=24:00:00

module load slurm_setup
export OMP_NUM_THREADS=28
mpiexec -n $SLURM_NTASKS ./hubbard2d.exe

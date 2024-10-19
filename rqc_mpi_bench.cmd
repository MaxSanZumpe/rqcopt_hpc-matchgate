#!/bin/bash
#SBATCH -J mpi_bench
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=01:00:00

module load slurm_setup
module load intel-oneapi-vtune
export OMP_NUM_THREADS=28
mpiexec -n $SLURM_NTASKS aps -result-dir=../benchmark/prof/mpi/task4_q16/ ./benchmark_mpi.exe

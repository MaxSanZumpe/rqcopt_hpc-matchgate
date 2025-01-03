#!/bin/bash
#SBATCH -J rqc_bench
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=112
# 224 is the maximum reasonable value for CoolMUC-4
#SBATCH --export=NONE
#SBATCH --time=05:00:00

module load slurm_setup
module load intel
module load intel-mpi
module load intel-mkl
module load hdf5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
./benchmark_threads

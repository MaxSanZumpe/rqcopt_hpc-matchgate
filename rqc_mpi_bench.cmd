#!/bin/bash
#SBATCH -J mpi_bench
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --export=NONE
#SBATCH --time=08:00:00

module load slurm_setup
module load intel-mpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpiexec -n $SLURM_NTASKS ./benchmark_mpi

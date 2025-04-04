#!/bin/bash
#SBATCH -J mpi_bench
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --export=NONE
#SBATCH --time=06:00:00

module load slurm_setup
module load intel-mpi
module load intel-mkl
module load hdf5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpiexec -n $SLURM_NTASKS ./benchmark_mpi

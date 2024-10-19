#!/bin/bash
#SBATCH -J rqc_bench
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=28
# 56 is the maximum reasonable value for CoolMUC-2
#SBATCH --export=NONE
#SBATCH --time=02:00:00

module load slurm_setup
module load intel-oneapi-vtune
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
aps -result-dir=../benchmark/prof/%x/ ./benchmark_multithreading.exe

#!/bin/bash
#SBATCH -J rqc_serial
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./build/
#SBATCH --mail-user=max.sanchez-zumpe@tum.de
#SBATCH --mail-type=ALL
#SBATCH --get-user-env
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --mem=800mb
#SBATCH --nodes=1-1
#SBATCH --export=NONE
#SBATCH --time=02:00:00
module load slurm_setup
 
./benchmark_serial.exe

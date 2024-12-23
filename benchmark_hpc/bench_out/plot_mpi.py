import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

nqubits = 12
nlayers = 5
ulayers = 253

script_dir = os.path.dirname(__file__)



script_dir = os.path.dirname(__file__)
file_list = glob.glob(f"mpi*n5*.hdf5", root_dir=script_dir)

file0 = os.path.join(script_dir, "n5_q16_u253_th112_110_threads_bench_matchgate.hdf5")



with h5py.File(os.path.join(script_dir, file0), "r") as f:
    t0 = f.attrs["Walltime"]

print(f"1 Node  t: {t0}s")

wtime = []
threads = []
tasks = []

for file in file_list:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime.append(f.attrs["Walltime"])
        threads.append(f.attrs["NUM_THREADS"])
        tasks.append(f.attrs["NUM_TASKS"])

for i in range(len(file_list)):
    print(f"Tasks: {tasks[i]}, threads: {threads[i]}, wtime: {wtime[i]} s")



import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

nqubits = 16
nlayers = 5
ulayers = 253

script_dir = os.path.dirname(__file__)


script_dir = os.path.dirname(__file__)
file_list = glob.glob(f"mpi/mpi*n5*.hdf5", root_dir=script_dir)

file_list2 = glob.glob(f"q{nqubits}/n5_q16_u253_th*_110_threads_bench_matchgate.hdf5", root_dir=script_dir)


wtime = []
threads = []
tasks = []


for file in file_list2:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime.append(f.attrs["Walltime"])
        threads.append(f.attrs["NUM_THREADS"])
        tasks.append(1)


for file in file_list:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime.append(f.attrs["Walltime"])
        threads.append(f.attrs["NUM_THREADS"])
        tasks.append(f.attrs["NUM_TASKS"])


threads = np.array(threads)
wtime = np.array(wtime)

cores = threads*tasks

xy = zip(cores, wtime)

xy1_sorted = sorted(xy, key = lambda pair: pair[0])

cores_sorted, wtime_sorted = zip(*xy1_sorted) 

for i in range(len(threads)):
    print(f"Tasks: {tasks[i]}, threads: {threads[i]}, wtime: {wtime[i]} s")

cores = np.array(cores_sorted)
wtime = np.array(wtime_sorted)

fig, ax = plt.subplots()
ax.scatter(cores, wtime[0]/wtime)
ax.plot([cores[0], cores[-1]], [1, cores[-1]/cores[0]])
fig.savefig(f"{script_dir}/mpi/plots/mpi_n{nlayers}_u{ulayers}")


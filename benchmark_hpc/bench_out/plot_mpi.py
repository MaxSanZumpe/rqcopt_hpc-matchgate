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

file0 = os.path.join(script_dir, f"q{nqubits}","n5_q16_u601_th112_110_threads_bench_matchgate.hdf5")
file1 = os.path.join(script_dir, f"q{nqubits}","n5_q16_u253_th56_110_threads_bench_matchgate.hdf5")



with h5py.File(os.path.join(script_dir, file0), "r") as f:
    t0 = f.attrs["Walltime"]


with h5py.File(os.path.join(script_dir, file1), "r") as f:
    t1 = f.attrs["Walltime"]


print(f"1 Node; 56 threads:{t1}s")
print(f"1 Node; 112 threads: {t0}s")

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


threads = np.array(threads)
wtime = np.array(wtime)

cores = threads*tasks

xy = zip(cores, wtime)

xy1_sorted = sorted(xy, key = lambda pair: pair[0])

cores_sorted, wtime_sorted = zip(*xy1_sorted) 

cores = np.array(cores_sorted)
wtime = np.array(wtime_sorted)

fig, ax = plt.subplots()
ax.scatter(cores, wtime[0]/wtime)
ax.plot([112, 112*4], [1, 4])
fig.savefig(f"{script_dir}/mpi/plots/mpi_n{nlayers}_u{ulayers}")


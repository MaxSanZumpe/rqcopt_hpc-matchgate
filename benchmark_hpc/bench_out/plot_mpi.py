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
file_list = glob.glob(f"mpi/mpi*n5*q16*th56*.hdf5", root_dir=script_dir)
file_list3 = glob.glob(f"mpi/mpi*n5*q16*th112*.hdf5", root_dir=script_dir)

file_list2 = glob.glob(f"q{nqubits}/n5_q16_u253_th*_110_threads_bench_matchgate.hdf5", root_dir=script_dir)

wtime = []
threads = []
tasks = []

wtime2 = []
threads2 = []
tasks2 = []

wtime3 = []
threads3 = []
tasks3 = []

for file in file_list:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime.append(f.attrs["Walltime"])
        threads.append(f.attrs["NUM_THREADS"])
        tasks.append(f.attrs["NUM_TASKS"])


for file in file_list3:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime3.append(f.attrs["Walltime"])
        threads3.append(f.attrs["NUM_THREADS"])
        tasks3.append(f.attrs["NUM_TASKS"])


for file in file_list2:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime2.append(f.attrs["Walltime"])
        threads2.append(f.attrs["NUM_THREADS"])
        tasks2.append(1)


print("OpenMP:")
for i in range(len(threads2)):
    print(f"Tasks: {tasks2[i]}, threads: {threads2[i]}, wtime: {wtime2[i]} s")

print("MPI BENCH:")
for i in range(len(threads)):
    print(f"Tasks: {tasks[i]}, threads: {threads[i]}, wtime: {wtime[i]} s")

for i in range(len(threads3)):
    print(f"Tasks: {tasks3[i]}, threads: {threads3[i]}, wtime: {wtime3[i]} s")

threads = np.array(threads)
wtime = np.array(wtime)
cores = threads*tasks
xy = zip(cores, wtime)
xy1_sorted = sorted(xy, key = lambda pair: pair[0])
cores_sorted, wtime_sorted = zip(*xy1_sorted) 
cores = np.array(cores_sorted)
wtime = np.array(wtime_sorted)


threads2 = np.array(threads2)
wtime2 = np.array(wtime2)
cores2 = threads2*tasks2
xy2 = zip(cores2, wtime2)
xy2_sorted = sorted(xy2, key = lambda pair: pair[0])
cores_sorted2, wtime_sorted2 = zip(*xy2_sorted) 
cores2 = np.array(cores_sorted2)
wtime2 = np.array(wtime_sorted2)


threads3 = np.array(threads3)
wtime3 = np.array(wtime3)
cores3 = threads3*tasks3
xy3 = zip(cores3, wtime3)
xy3_sorted = sorted(xy3, key = lambda pair: pair[0])
cores_sorted3, wtime_sorted3 = zip(*xy3_sorted) 
cores3 = np.array(cores_sorted3)
wtime3 = np.array(wtime_sorted3)

fig, ax = plt.subplots()

ax.scatter(cores2, wtime2[0]/wtime2, marker=".", color = "red", label = "OpenMP")
ax.scatter(cores , wtime2[0]/wtime, marker=".", color = "black", label = "MPI;  56 threads/task")
ax.scatter(cores3, wtime2[0]/wtime3, marker=".", color = "blue", label = "MPI; 112 threads/task")
ax.plot([56, 448], [1, 8], color = "green", label = "Ideal scaling")

ax.set_title(f"MPI benchmark (q = {nqubits}; n = {nlayers})", fontsize= 12)

ax.set_xlabel("Total threads", fontsize = 12)
ax.set_ylabel("Wall time speed-up", fontsize = 12)

ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi/plots/mpi_n{nlayers}_u{ulayers}", dpi=300)


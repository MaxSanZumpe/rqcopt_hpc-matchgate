import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

nqubits = 16
nlayers = 5
ulayers = 253

script_dir = os.path.dirname(__file__)

script_dir = os.path.dirname(__file__)
file_list7 = glob.glob(f"q{nqubits}/n5_q16_u253_th*_110_threads_bench_matchgate.hdf5", root_dir=script_dir)
file_list1 = glob.glob(f"q{nqubits}/n5_q16_u601_th*_110_threads_bench_matchgate.hdf5", root_dir=script_dir)

file_list2 = glob.glob(f"mpi/mpi*n5*q16_u253*th14_*.hdf5", root_dir=script_dir)
file_list3 = glob.glob(f"mpi/mpi*n5*q16_u253*th28_*.hdf5", root_dir=script_dir)
file_list4 = glob.glob(f"mpi/mpi*n5*q16_u253*th56_*.hdf5", root_dir=script_dir)
file_list5 = glob.glob(f"mpi/mpi*n5*q16_u253*th112_*.hdf5", root_dir=script_dir)
file_list6 = glob.glob(f"mpi/mpi*n5*q16_u253*th2_*.hdf5", root_dir=script_dir)

mapping = ["omp", 56, 112, 28, 14, 2]
file_arr = [file_list1, file_list2, file_list3, file_list4, file_list5, file_list6, file_list7]

wtime_arr = []
threads_arr = []
cores_arr = []
tasks_arr = []

for aux, file_list in enumerate(file_arr):
    
    wtime = []
    threads = []
    tasks = []
    for file in file_list:
        with h5py.File(os.path.join(script_dir, file), "r") as f:
            wtime.append(f.attrs["Walltime"])
            threads.append(f.attrs["NUM_THREADS"])
            if aux != 0 and aux != 6:
                tasks.append(f.attrs["NUM_TASKS"])
            else:
                tasks.append(1)

    threads = np.array(threads)
    wtime = np.array(wtime)
    cores = threads*tasks
    
    xy = zip(cores, wtime, tasks)
    xy1_sorted = sorted(xy, key = lambda pair: pair[0])
    cores_sorted, wtime_sorted, tasks_sorted = zip(*xy1_sorted) 
    cores = np.array(cores_sorted)
    wtime = np.array(wtime_sorted)
    tasks = np.array(tasks_sorted)

    wtime_arr.append(wtime)
    threads_arr.append(threads)
    cores_arr.append(cores)
    tasks_arr.append(tasks)

print("OMP BENCH:")
print(f"    Threads: {cores_arr[0]} -> Wtime: {wtime_arr[0]}")
print(f"    Threads: {cores_arr[6]} -> Wtime: {wtime_arr[6]}")



print("MPI BENCH:")
for i in range(1, len(file_arr)):
    print(f"{threads_arr[i][0]} threads per task:")
    print(f"    Tasks: {tasks_arr[i]}; Cores: {cores_arr[i]} -> Wtime: {wtime_arr[i]}")

print(wtime_arr[4])

fig, ax = plt.subplots()

#ax.scatter([4]   ,   wtime_arr[0][4]/wtime_arr[4], marker = "o", color = "red", label= "MPI; 112 threads/task")
ax.scatter([1, 2, 4], wtime_arr[2][0]/wtime_arr[2], color = "black", label= "MPI; 28 threads/task")
#ax.scatter([1, 2, 4], wtime_arr[3][0]/wtime_arr[3], color = "black", label= "Benchmarks")

ax.plot([1, 4], [1, 4], color = "green", label = "Ideal scaling")

ax.set_title(f"Node scaling (q = {nqubits}; n = {nlayers})", fontsize= 12)

ax.set_xlabel("Compute nodes", fontsize = 12)
ax.set_ylabel("Wall time speed-up", fontsize = 12)
ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4]))

ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi/plots/mpi_n{nlayers}_u{ulayers}_tasks.png", dpi=300)



fig, ax = plt.subplots()

c0 = 14
ax.scatter(c0, 1, color = "black")
ax.plot(cores_arr[1], wtime_arr[0][0]/wtime_arr[1], marker="o", color = "orange", label = "14 threads/task")
ax.plot(cores_arr[2], wtime_arr[0][0]/wtime_arr[2], marker="o", color = "purple", label = "28 threads/task")
ax.plot(cores_arr[3], wtime_arr[0][0]/wtime_arr[3], marker="o", color = "black", label = "56 threads/task")
ax.plot(cores_arr[4], wtime_arr[0][0]/wtime_arr[4], marker="o", color = "red", label = "112 threads/task")
ax.plot(cores_arr[5], wtime_arr[0][0]/wtime_arr[5], marker="o", color = "blue", label = "  2 threads/task")


ax.plot([c0, cores_arr[2][-1]], [1, 448/c0], color = "green", label = "Ideal scaling")

ax.set_title(f"MPI scaling (q = {nqubits}; n = {nlayers})", fontsize= 12)

ax.set_xlabel("Total threads", fontsize = 12)
ax.set_ylabel("Wall time speed-up", fontsize = 12)
ax.xaxis.set_major_locator(FixedLocator([14, 56, 112, 224, 448]))

ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi/plots/mpi_n{nlayers}_u{ulayers}", dpi=300)


fig, ax = plt.subplots()
ax.plot(cores_arr[0], wtime_arr[0][0]/wtime_arr[0], marker=".", color = "black", label = "OpenMP")
ax.plot([14, 112], [1, 8], color = "green", label = "Ideal scaling")

ax.scatter(cores_arr[3][0], wtime_arr[0][0]/wtime_arr[3][0], marker="*", color = "red" ,  label = "MPI; 56 threads/task")
ax.scatter(cores_arr[2][0], wtime_arr[0][0]/wtime_arr[2][0], marker="^", color = "purple",label = "MPI; 28 threads/task")


ax.set_title(f"MPI vs. OpenMP (q = {nqubits}; n = {nlayers})", fontsize= 12)

ax.set_xlabel("Total threads", fontsize = 12)
ax.set_ylabel("Wall time speed-up", fontsize = 12)

ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi/plots/openmp16_n{nlayers}_u{ulayers}", dpi=300)


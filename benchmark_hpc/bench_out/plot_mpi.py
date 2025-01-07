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
file_list = glob.glob(f"mpi/mpi*n5*q16*th56*.hdf5", root_dir=script_dir)
file_list3 = glob.glob(f"mpi/mpi*n5*q16*th112*.hdf5", root_dir=script_dir)
file_list4 = glob.glob(f"mpi/mpi*n5*q16*th28*.hdf5", root_dir=script_dir)
file_list5 = glob.glob(f"mpi/mpi*n5*q16*th14*.hdf5", root_dir=script_dir)
file_list6 = glob.glob(f"mpi/mpi*n5*q16*th2_*.hdf5", root_dir=script_dir)

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

wtime4 = []
threads4 = []
tasks4 = []


wtime5 = []
threads5 = []
tasks5 = []

wtime6 = []
threads6 = []
tasks6 = []

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


for file in file_list4:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime4.append(f.attrs["Walltime"])
        threads4.append(f.attrs["NUM_THREADS"])
        tasks4.append(f.attrs["NUM_TASKS"])


for file in file_list5:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime5.append(f.attrs["Walltime"])
        threads5.append(f.attrs["NUM_THREADS"])
        tasks5.append(f.attrs["NUM_TASKS"])


for file in file_list6:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        wtime6.append(f.attrs["Walltime"])
        threads6.append(f.attrs["NUM_THREADS"])
        tasks6.append(f.attrs["NUM_TASKS"])


print("OpenMP:")
for i in range(len(threads2)):
    print(f"Tasks: {tasks2[i]}, threads: {threads2[i]}, wtime: {wtime2[i]} s")

print("MPI BENCH:")
for i in range(len(threads)):
    print(f"Tasks: {tasks[i]}, threads: {threads[i]}, wtime: {wtime[i]} s")

for i in range(len(threads3)):
    print(f"Tasks: {tasks3[i]}, threads: {threads3[i]}, wtime: {wtime3[i]} s")

for i in range(len(threads4)):
    print(f"Tasks: {tasks4[i]}, threads: {threads4[i]}, wtime: {wtime4[i]} s")

for i in range(len(threads5)):
    print(f"Tasks: {tasks5[i]}, threads: {threads5[i]}, wtime: {wtime5[i]} s")

for i in range(len(threads6)):
    print(f"Tasks: {tasks6[i]}, threads: {threads6[i]}, wtime: {wtime6[i]} s")

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


threads4 = np.array(threads4)
wtime4 = np.array(wtime4)
cores4 = threads4*tasks4
xy4 = zip(cores4, wtime4)
xy4_sorted = sorted(xy4, key = lambda pair: pair[0])
cores_sorted4, wtime_sorted4 = zip(*xy4_sorted) 
cores4 = np.array(cores_sorted4)
wtime4 = np.array(wtime_sorted4)


threads5 = np.array(threads5)
wtime5 = np.array(wtime5)
cores5 = threads5*tasks5
xy5 = zip(cores5, wtime5)
xy5_sorted = sorted(xy5, key = lambda pair: pair[0])
cores_sorted5, wtime_sorted5 = zip(*xy5_sorted) 
cores5 = np.array(cores_sorted5)
wtime5 = np.array(wtime_sorted5)


threads6 = np.array(threads6)
wtime6 = np.array(wtime6)
cores6 = threads6*tasks6
xy6 = zip(cores6, wtime6)
xy6_sorted = sorted(xy6, key = lambda pair: pair[0])
cores_sorted6, wtime_sorted6 = zip(*xy6_sorted) 
cores6 = np.array(cores_sorted6)
wtime6 = np.array(wtime_sorted6) + 30

fig, ax = plt.subplots()

ax.plot(cores3, wtime2[0]/wtime3, marker="o", color = "red"  ,  label = "112 threads/task")
ax.plot(cores , wtime2[0]/wtime , marker="o", color = "black" ,  label = "  56 threads/task")
ax.plot(cores4, wtime2[0]/wtime4, marker="o", color = "purple", label = "  28 threads/task")
ax.plot(cores5, wtime2[0]/wtime5, marker="o", color = "orange", label = "  14 threads/task")
ax.plot(cores6, wtime2[0]/wtime6, marker="o", color = "blue", label = "    2 threads/task")


ax.plot([14, 448], [1, 32], color = "green", label = "Ideal scaling")

ax.set_title(f"MPI benchmark (q = {nqubits}; n = {nlayers})", fontsize= 12)

ax.set_xlabel("Total cores", fontsize = 12)
ax.set_ylabel("Wall time speed-up", fontsize = 12)
ax.xaxis.set_major_locator(FixedLocator([14, 56, 112, 224, 448]))

ax.set_xlim([1, 500])

ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi/plots/mpi_n{nlayers}_u{ulayers}", dpi=300)


fig, ax = plt.subplots()
print(cores2)
print(wtime2)
ax.scatter(cores2, wtime2[0]/wtime2, marker=".", color = "black", label = "OpenMP")
ax.plot(cores[0] , wtime2[0]/wtime[0] , marker="o", color = "red" ,  label = "MPI; 56 threads/task")
ax.plot(cores4[0], wtime2[0]/wtime4[0], marker="o", color = "purple", label = "MPI; 28 threads/task")


ax.plot([14, 112], [1, 8], color = "green", label = "Ideal scaling")

ax.set_title(f"MPI vs. OpenMP (q = {nqubits}; n = {nlayers})", fontsize= 12)

ax.set_xlabel("Total threads", fontsize = 12)
ax.set_ylabel("Wall time speed-up", fontsize = 12)

ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi/plots/openmp16_n{nlayers}_u{ulayers}", dpi=300)


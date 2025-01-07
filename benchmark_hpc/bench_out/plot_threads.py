import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nqubits = 12
nlayers = 5
ulayers = 601

script_dir = os.path.dirname(__file__)
data_dir   = os.path.join(script_dir, f"q{nqubits}")


file_list1 = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_th*_110_*threads*.hdf5")
file_list2 = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_th*_010_*threads*.hdf5")

if nqubits != 16 and nqubits != 14:
    file_list1.append(glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_1*serial*.hdf5")[0])
    file_list2.append(glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_0*serial*.hdf5")[0])

critical = False
if critical:
    file_list3 = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_th*_010_critical_matchgate*.hdf5")
    file_list3.append(glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_0*serial*.hdf5")[0])

mpi = True
if mpi:
    file4 = glob.glob(f"{script_dir}/mpi/mpi*n5*q12*tasks4*.hdf5")[0]
    with h5py.File(file4, "r") as f:
        wtime_mpi = f.attrs["Walltime"]
        threads_mpi=f.attrs["NUM_THREADS"]
        tasks = f.attrs["NUM_TASKS"]
        cores = tasks*threads_mpi


wtime1 = []
threads1 = []

wtime2 = []
threads2 = []

for file1, file2 in zip(file_list1, file_list2):
    with h5py.File(file1, "r") as f:
        wtime1.append(f.attrs["Walltime"])
        threads1.append(f.attrs["NUM_THREADS"])

    with h5py.File(file2, "r") as f:
        wtime2.append(f.attrs["Walltime"])
        threads2.append(f.attrs["NUM_THREADS"])


if critical: 
    wtime3 = []
    threads3 = []

    for file in file_list3:
        with h5py.File(file, "r") as f:
            wtime3.append(f.attrs["Walltime"])
            threads3.append(f.attrs["NUM_THREADS"])
    
    xy = zip(threads3, wtime3)
    xy_sorted = sorted(xy, key = lambda pair: pair[0])
    threads3_sorted, wtime3_sorted = zip(*xy_sorted) 

    wtime3 = np.array(wtime3_sorted)
    threads3 = np.array(threads3_sorted)
    
    
xy1 = zip(threads1, wtime1)
xy2 = zip(threads2, wtime2)

xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
xy2_sorted = sorted(xy2, key = lambda pair: pair[0])

threads1_sorted, wtime1_sorted = zip(*xy1_sorted) 
threads2_sorted, wtime2_sorted = zip(*xy2_sorted) 

wtime1 = np.array(wtime1_sorted)
wtime2 = np.array(wtime2_sorted)

threads1 = np.array(threads1_sorted)
threads2 = np.array(threads2_sorted)

for a, b, c in zip(threads1, wtime1, wtime2):
    print(f"threads = {a} -> wtime1: {b}s -> wtime2: {c}s")

if mpi:
    print(cores, wtime_mpi)

fig, ax = plt.subplots()

ax.scatter(threads1, wtime1[0]/wtime1, marker=".", label = "$T_{parallel}/T_{serial}$", color = "black")

if critical: 
    ax.scatter(threads3, wtime3[0]/wtime3, marker=".", label = "No critical sections", color = "blue")

if mpi:
    ax.scatter(cores, wtime1[0]/wtime_mpi)

print(file4)

ax.plot(threads1, threads1, label = "Ideal scaling", color = "green")

ax.tick_params(axis="x", labelsize = 12)
ax.tick_params(axis="y", labelsize = 12)


ax.set_xlabel("Threads", fontsize = 20)
ax.set_ylabel("Wall time speed-up", fontsize = 20)

#ax.set_title(f"Multithreading benchmark: q = {nqubits}, n = {nlayers}")

ax.legend(fontsize=16)
fig.savefig(f"{data_dir}/plots/n{nlayers}_q{nqubits}_u{ulayers}_thread_scaling.png", dpi=400)



    
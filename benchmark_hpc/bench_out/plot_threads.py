import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nqubits = 12
nlayers = 5
ulayers = 253

script_dir = os.path.dirname(__file__)
data_dir   = os.path.join(script_dir, f"n{nlayers}_q{nqubits}_u{ulayers}_threads_bench_matchgate")

file_list1 = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_th*_0*.hdf5")
file_list2 = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_th*_1*.hdf5")


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


xy1 = zip(threads1, wtime1)

xy1_sorted = sorted(xy1, key = lambda pair: pair[0])

threads1_sorted, wtime1_sorted = zip(*xy1_sorted) 

wtime1 = np.array(wtime1_sorted)
wtime2 = np.array(wtime2)

threads1 = np.array(threads1_sorted)
threads2 = np.array(threads2)


fig, ax = plt.subplots()

ax.scatter(threads1, wtime1[0]/wtime1, marker=".", label = "$T_{parallel}/T_{serial}$", color = "black")
ax.plot(threads1, threads1, label = "Ideal scaling", color = "green")

ax.set_xlabel("Thread number")
ax.set_ylabel("Speed up")

ax.set_title(f"Qubits = {nqubits}, Layers = {nlayers}")

ax.legend()
fig.savefig(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_scaling.png")


    
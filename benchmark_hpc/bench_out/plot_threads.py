import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nqubits = 12
nlayers = 7
ulayers = 601

script_dir = os.path.dirname(__file__)
data_dir   = os.path.join(script_dir, f"q{nqubits}")


file_list1 = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_th*_010*.hdf5")
file_list2 = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_th*_110*.hdf5")

if nqubits != 16:
    file_list1.append(glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_0*serial*.hdf5")[0])
    file_list2.append(glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u{ulayers}_1*serial*.hdf5")[0])

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

fig, ax = plt.subplots()

ax.scatter(threads1, wtime1[0]/wtime1, marker=".", label = "$T_{parallel}/T_{serial}$", color = "black")
ax.plot(threads1, threads1, label = "Ideal scaling", color = "green")
ax.plot([56, 56], [1, 4])

ax.set_xlabel("Thread number")
ax.set_ylabel("Walltime speed-up")

ax.set_title(f"Qubits = {nqubits}, Layers = {nlayers}")

ax.legend()
fig.savefig(f"{data_dir}/plots/n{nlayers}_q{nqubits}_u{ulayers}_thread_scaling.png")


fig, ax = plt.subplots()

#ax.scatter(threads1, wtime1, marker=".", label = "Matchgates", color = "black")
ax.scatter(threads2, (wtime1 - wtime2)/wtime1, marker=".", label = "Matchgates + Invariance", color = "green")


ax.set_xlabel("Thread number")
ax.set_ylabel("Walltime (s)")

ax.set_title(f"Qubits = {nqubits}, Layers = {nlayers}")

ax.legend()
fig.savefig(f"{data_dir}/plots/n{nlayers}_q{nqubits}_u{ulayers}_invariance_scaling.png")


    
import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nqubits = 12
ulayers = 601

script_dir = os.path.dirname(__file__)
data_dir   = os.path.join(script_dir, f"q{nqubits}")

file_list1 = glob.glob(f"{data_dir}/n*_q{nqubits}_u{ulayers}_th112_010_*threads*.hdf5")
file_list2 = glob.glob(f"{data_dir}/n*_q{nqubits}_u{ulayers}_th112_110_*threads*.hdf5")


wtime1 = []
layers1 = []

wtime2 = []
layers2 = []

for file1, file2 in zip(file_list1, file_list2):
    with h5py.File(file1, "r") as f:
        wtime1.append(f.attrs["Walltime"])
        layers1.append(f.attrs["nlayers"])

    with h5py.File(file2, "r") as f:
        wtime2.append(f.attrs["Walltime"])
        layers2.append(f.attrs["nlayers"])

    
xy1 = zip(layers1, wtime1)
xy2 = zip(layers2, wtime2)

xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
xy2_sorted = sorted(xy2, key = lambda pair: pair[0])

layers1_sorted, wtime1_sorted = zip(*xy1_sorted) 
layers2_sorted, wtime2_sorted = zip(*xy2_sorted) 

wtime1 = np.array(wtime1_sorted)
wtime2 = np.array(wtime2_sorted)

layers1 = np.array(layers1_sorted)
layers2 = np.array(layers2_sorted)

fig, ax = plt.subplots()

p1 = np.polyfit(layers1, wtime1, 2)
p2 = np.polyfit(layers2, wtime2, 1)

ax.scatter(layers1, wtime1, marker=".", color="black", label = "No invariance")
ax.scatter(layers2, wtime2, marker="x", color="red", label = "With Invarinace")

ax.plot(layers1, p1[0]*layers1**2 + p1[1]*layers1 + p1[2])
ax.plot(layers2, p2[0]*layers2 + p2[1])


ax.set_xlabel("Circuit Layers")
ax.set_ylabel("Walltime (s)")

ax.set_title(f"Circuit Layers benchmark for q = {nqubits})")

ax.legend()
fig.savefig(f"{data_dir}/plots/q{nqubits}_u{ulayers}_layers_scaling.png")



    
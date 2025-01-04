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

for file1 in file_list1:
    with h5py.File(file1, "r") as f:
        wtime1.append(f.attrs["Walltime"])
        layers1.append(f.attrs["nlayers"])

for file2 in file_list2:
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

print(layers1, layers2)

p1 = np.polyfit(layers1, wtime1, 2)
p2 = np.polyfit(layers2, wtime2, 1)

print(p1)
print(p2)

s = f"{p2[0]:.2f}$n$+{np.abs(p2[1]):.1f}"
z = f"{p1[0]:.2f}$n^2$+{np.abs(p1[1]):.2f}$n$ + {np.abs(p1[2]):.0f}"

ax.scatter(layers1, wtime1, marker=".", color="blue", label = f"Without invariance: {z}")
ax.scatter(layers2, wtime2, marker="x", color="red", label = f"With invarinace: {s}")

ax.plot(layers1, p1[0]*layers1**2 + p1[1]*layers1 + p1[2])
ax.plot(layers2, p2[0]*layers2 + p2[1])


ax.set_xlabel("Circuit Layers", fontsize = 12)
ax.set_ylabel("Wall time (s)", fontsize = 12)

ax.set_title(f"Circuit layer scaling becnhmark (q = {nqubits})")

ax.legend(fontsize = 11)
fig.savefig(f"{data_dir}/plots/q{nqubits}_u{ulayers}_layers_scaling.png", dpi=300)



    
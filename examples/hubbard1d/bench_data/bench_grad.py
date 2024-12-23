import h5py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

n = 5
q = 12

script_dir = os.path.dirname(__file__)
file_list = glob.glob(f"*q{q}*_grad*.hdf5", root_dir=script_dir)

print(file_list)

wtime = []

layers = []

for file in file_list:
    with h5py.File(os.path.join(script_dir, file), "r") as f:
        
        wtime.append(f.attrs["Walltime"])
        layers.append(f.attrs["nlayers"])

wtime = np.array(wtime)
layers = np.array(layers)

p = np.polyfit(layers, wtime, 1)


fig, ax = plt.subplots()
ax.scatter(layers, wtime, marker=".")
ax.set_xlabel("Layers")
ax.plot(layers, p[0]*layers + p[1])
ax.set_ylabel("Runtime wall clock s")
ax.set_title(f"Qubits: {q} Scaling: {p[0]}n")

plt.savefig(os.path.join(script_dir, "bench_grad_match2"))




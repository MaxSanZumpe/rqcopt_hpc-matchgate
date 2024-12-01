import h5py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, "data")
file_list = glob.glob(f"*n*_q8_*010.hdf5", root_dir=data_dir)

print(file_list)

costs = []
layers  = []
start = []

for file in file_list:
    with h5py.File(os.path.join(data_dir, file), "r") as f:
        
        start.append(np.array(f["f_iter"])[0])
        costs.append(np.array(f["f_iter"])[-1])
        layers.append(f.attrs["nlayers"])

start = 2*np.array(start) + 2**9
norms = 2*np.array(costs) + 2**9

fig, ax = plt.subplots()
ax.scatter(layers, start, label = "trotter")
ax.scatter(np.array(layers), norms, label = "opt")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

plt.savefig(os.path.join(data_dir, "norm_error"))




import h5py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

n = 5

script_dir = os.path.dirname(__file__)
file_list = glob.glob(f"*hess*n{n}*.hdf5", root_dir=script_dir)

print(file_list)

wtime0 = []
wtime1 = []

qubits = []

for file in file_list:
    with h5py.File(os.path.join(script_dir, file), "r") as f:
        
        wtime0.append(f.attrs["Walltime_inv0"])
        wtime1.append(f.attrs["Walltime_inv1"])
        qubits.append(f.attrs["nqubits"])

fig, ax = plt.subplots()
ax.scatter(qubits, wtime0, label = "no_inv")
ax.scatter(qubits, wtime1, label = "inv")
#ax.set_xscale("log")
ax.set_title("layers = {n} | general gates")
ax.set_yscale("log")
ax.legend()

plt.savefig(os.path.join(script_dir, "bench_hess_match1"))




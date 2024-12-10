import h5py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

n = 5

script_dir = os.path.dirname(__file__)
file_list = glob.glob(f"*n{n}*_grad*.hdf5", root_dir=script_dir)

print(file_list)

wtime = []

qubits = []

for file in file_list:
    with h5py.File(os.path.join(script_dir, file), "r") as f:
        
        wtime.append(f.attrs["Walltime"])
        qubits.append(f.attrs["nqubits"])

wtime = np.array(wtime)
qubits = np.array(qubits)
p = np.polyfit(qubits, np.log(wtime), 1)

print(p[0])

fig, ax = plt.subplots()
ax.semilogy(qubits, wtime, ".")
ax.plot(qubits, np.exp((p[0]*qubits + p[1])))
ax.set_title(f"layers = {n} | general gates")

plt.savefig(os.path.join(script_dir, "bench_grad_match1"))




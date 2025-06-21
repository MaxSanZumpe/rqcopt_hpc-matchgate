import h5py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

n = 5
q = 12

script_dir = os.path.dirname(__file__)
file_list = glob.glob(f"hubbard1d_grad_bench_n*_q{q}_u589.hdf5", root_dir=script_dir)
file_list2 = glob.glob(f"hubbard1d_grad_bench_n21_q*_u589.hdf5", root_dir=script_dir)

wtime = []
layers = []

for file in file_list:
    print(file)
    with h5py.File(os.path.join(script_dir, file), "r") as f:
        
        wtime.append(f.attrs["Walltime"])
        layers.append(f.attrs["nlayers"])

wtime = np.array(wtime)
layers = np.array(layers)

p = np.polyfit(layers, wtime, 1)


fig, ax = plt.subplots()
ax.scatter(layers, wtime, marker=".", s=300, color="black")
ax.set_xlabel("Number of layers", fontsize=20)
ax.plot((5, 21), (p[0]*5 + p[1], p[0]*21 + p[1]),
        color="blue", linewidth=3, alpha = 0.4,
        label=f"{p[0]:.4f}$n_{{layers}}$ - {np.abs(p[1]):.4f}")
ax.set_ylabel("Evaluation time (s)", fontsize=20)
ax.legend(fontsize=14, loc="lower right")
ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "grad_bench_layers_c_code.pdf"))



wtime2 = []
qubits = []

for file in reversed(file_list2):
    print(file)
    with h5py.File(os.path.join(script_dir, file), "r") as f:
        
        wtime2.append(f.attrs["Walltime"])
        qubits.append(f.attrs["nqubits"])

wtime2 = np.array(wtime2)
qubits = np.array(qubits)

p = np.polyfit(qubits, np.log2(wtime2), 1)

fig, ax = plt.subplots()
ax.scatter(qubits, wtime2, marker=".", s=300, color="black")
ax.set_xlabel("Number of qubits", fontsize=20)
ax.plot((8, 14), (2**(p[0]*8 + p[1]), 2**(p[0]*14 + p[1])),
        color="blue", linewidth=3, alpha = 0.4,
        label=f"{p[0]:.2f}*$2^{{q}}$ - {np.abs(p[1]):.2f}")
ax.set_ylabel("Evaluation time (s)", fontsize=20)
ax.legend(fontsize=14, loc="lower right")
ax.set_yscale('log')
ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "grad_bench_qubits_c_code.pdf"))

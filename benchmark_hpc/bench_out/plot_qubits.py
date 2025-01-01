import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nlayers = 5
ulayers = 601

script_dir = os.path.dirname(__file__)


wtime1 = []
qubits1 = []

wtime2 = []
qubits2 = []

for q in range(8,12,2):
    data_dir   = os.path.join(script_dir, f"q{q}")
    file_list1 = glob.glob(f"{data_dir}/n{nlayers}_q*_u{ulayers}_th112_010_*threads*.hdf5")
    file_list2 = glob.glob(f"{data_dir}/n{nlayers}*_q*_u{ulayers}_th112_110_*threads*.hdf5")

    for file1, file2 in zip(file_list1, file_list2):
        with h5py.File(file1, "r") as f:
            wtime1.append(f.attrs["Walltime"])
            qubits1.append(f.attrs["nqubits"])

        with h5py.File(file2, "r") as f:
            wtime2.append(f.attrs["Walltime"])
            qubits2.append(f.attrs["nqubits"])

    
xy1 = zip(qubits1, wtime1)
xy2 = zip(qubits2, wtime2)

xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
xy2_sorted = sorted(xy2, key = lambda pair: pair[0])

qubits1_sorted, wtime1_sorted = zip(*xy1_sorted) 
qubits2_sorted, wtime2_sorted = zip(*xy2_sorted) 

wtime1 = np.array(wtime1_sorted)
wtime2 = np.array(wtime2_sorted)

qubits1 = np.array(qubits1_sorted)
qubits2 = np.array(qubits2_sorted)

fig, ax = plt.subplots()

# p1 = np.polyfit(qubits1, wtime1, 2)
# p2 = np.polyfit(qubits2, wtime2, 1)

ax.scatter(qubits1, wtime1, marker=".", color="black", label = "No invariance")
#ax.scatter(qubits2, wtime2/wtime1[0], marker="x", color="red", label = "With Invarinace")

# ax.plot(qubits1, p1[0]*qubits1**2 + p1[1]*qubits1 + p1[2])
# ax.plot(qubits2, p2[0]*qubits2 + p2[1])

ax.set_xlabel("Qubits")
ax.set_ylabel("Walltime (s)")

ax.set_title(f"Qubit Walltime scaling (N = {nlayers})")

ax.legend()
fig.savefig(f"{data_dir}/plots/n{nlayers}_u{ulayers}_qubit_scaling.png")



    
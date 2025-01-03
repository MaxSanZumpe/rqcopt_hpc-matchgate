import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nlayers = 5
ulayers = 601

script_dir = os.path.dirname(__file__)

q1 = 8

wtime1 = []
qubits1 = []

wtime2 = []
qubits2 = []

for q in range(8,14,2):
    data_dir   = os.path.join(script_dir, f"q{q}")
    file_list1 = glob.glob(f"{data_dir}/n{nlayers}_q*_u{ulayers}_th112_110*.hdf5")
    file_list2 = glob.glob(f"{data_dir}/n{nlayers}_q*_u{ulayers}_th112_010*.hdf5")

    with h5py.File(file_list1[0], "r") as f:
        wtime1.append(f.attrs["Walltime"])
        qubits1.append(f.attrs["nqubits"])

    with h5py.File(file_list2[0], "r") as f:
        wtime2.append(f.attrs["Walltime"])
        qubits2.append(f.attrs["nqubits"])



data_dir   = os.path.join(script_dir, f"q{16}")
file_list1 = glob.glob(f"{data_dir}/n{nlayers}_q*_u{ulayers}_th112_110*.hdf5")
with h5py.File(file_list1[0], "r") as f:
    wtime1.append(f.attrs["Walltime"])
    qubits1.append(f.attrs["nqubits"])

print(qubits1)
print(wtime1)

xy1 = zip(qubits1, wtime1)
xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
qubits1_sorted, wtime1_sorted = zip(*xy1_sorted) 

wtime1 = np.array(wtime1_sorted)
qubits1 = np.array(qubits1_sorted)

xy2 = zip(qubits2, wtime2)
xy2_sorted = sorted(xy2, key = lambda pair: pair[0])
qubits2_sorted, wtime2_sorted = zip(*xy2_sorted) 

qubits2 = np.array(qubits2_sorted)
wtime2 = np.array(wtime2_sorted)

fig, ax = plt.subplots()

p = np.polyfit(qubits1, np.log(wtime1), 1)
print(f"Fit slope: {p[0]}")

x_fit = np.linspace(q1,17, 50)
fit = np.exp(np.polyval(p, x_fit))

ax.scatter(qubits1, wtime1, marker=".", color="black")
#ax.scatter(qubits2, wtime2, marker=".", color="black")

#ax.plot(x_fit, fit)


ax.set_xlabel("Qubits")
ax.set_ylabel("Walltime (s)")

ax.set_title(f"Qubit Walltime scaling (n = {nlayers})")

#ax.legend()
fig.savefig(f"{data_dir}/plots/n{nlayers}_u{ulayers}_qubit_scaling.png")



    
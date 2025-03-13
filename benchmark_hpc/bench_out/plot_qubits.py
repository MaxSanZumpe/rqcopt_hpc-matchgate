import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


nlayers = 5
ulayers = 601

script_dir = os.path.dirname(__file__)

q1 = 10

wtime1 = []
qubits1 = []

wtime2 = []
qubits2 = []

for q in range(8,18,2):
    data_dir   = os.path.join(script_dir, f"q{q}")
    file_list1 = glob.glob(f"{data_dir}/n{5}_q*_u{ulayers}_th112_010*.hdf5")
    if q != 16:
        file_list2 = glob.glob(f"{data_dir}/n{7}_q*_u{ulayers}_th112_010*.hdf5")

    with h5py.File(file_list1[0], "r") as f:
        wtime1.append(f.attrs["Walltime"])
        qubits1.append(f.attrs["nqubits"])

    with h5py.File(file_list2[0], "r") as f:
        wtime2.append(f.attrs["Walltime"])
        qubits2.append(f.attrs["nqubits"])


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

sft = 0
p = np.polyfit(qubits1[sft:], np.log2(wtime1[sft:]/60), 1)
print(f"Fit slope: {p[0]}; offset: {p[1]}")

def f(q, a, b, p0, p1, p2):
    return (p0*q**2 + p1*q + p2)*2**(a*q + b)


def f2(q, a, b):
    return 2**(a*q + b)

p0 = [p[0], p[1]]

popt, pcov = curve_fit(f2, qubits1[sft:], wtime1[sft:]/60, p0)
print(popt)

x_fit = np.linspace(8,16.2, 50)
fit1 = 2**(np.polyval(p, x_fit))
fit2 = f2(x_fit, popt[0], popt[1])

fig, ax = plt.subplots()

ax.scatter(qubits1[0:], wtime1[0:]/60, marker="^", color="red", label = "Benchmarks (n = 5)")
ax.set_yscale("log")

s = f"{p[0]:.2f}q-{np.abs(p[1]):.1f}"
print(s)

ax.plot(x_fit, fit1, color="black", label=f"Fit: $2^{{{s}}}$")
ax.plot(x_fit, fit2, color="blue")



ax.set_xlabel("Qubits", fontsize = 12)
ax.set_ylabel("Wall time (min)", fontsize = 12)

ax.set_title(f"Qubit scaling benchmark")

ax.legend(fontsize = 12)
fig.savefig(f"{data_dir}/plots/n{nlayers}_u{ulayers}_qubit_scaling.png", dpi = 300)



    
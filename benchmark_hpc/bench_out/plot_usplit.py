import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nqubits = 12
nlayers = 5

script_dir = os.path.dirname(__file__)
data_dir   = os.path.join(script_dir, f"q{nqubits}")

file_list = glob.glob(f"{data_dir}/n{nlayers}_q{nqubits}_u*_th112_010_*threads*.hdf5")

wtime = []
usplit = []

for file in file_list:
    with h5py.File(file, "r") as f:
        wtime.append(f.attrs["Walltime"])
        usplit.append(f.attrs["ulayers"])

xy = zip(usplit, wtime)
xy_sorted = sorted(xy, key = lambda pair: pair[0])
usplit_sorted, wtime_sorted = zip(*xy_sorted) 

wtime = np.array(wtime_sorted)
usplit = np.array(usplit_sorted)

print(usplit)

p = np.polyfit(usplit, wtime, 1)

fig, ax = plt.subplots()

ax.scatter(usplit, wtime, marker="^", color = "red", label = "Benchmarks")
ax.plot(usplit, p[0]*usplit + p[1], color = "black", label = f"Fit")

ax.set_xlabel("Layers")
ax.set_ylabel("Walltime (s)")

ax.set_title(f"Target unitary splitting banchmark (Q = 12)")
ax.legend()

fig.savefig(f"{data_dir}/plots/n{nlayers}_q{nqubits}_usplit_scaling.png")




    
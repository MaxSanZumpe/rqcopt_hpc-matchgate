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
file_list2 = glob.glob(f"{data_dir}/n{9}_q{nqubits}_u*_th112_010_*threads*.hdf5")


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


wtime2 = []
usplit2 = []

for file in file_list2:
    with h5py.File(file, "r") as f:
        wtime2.append(f.attrs["Walltime"])
        usplit2.append(f.attrs["ulayers"])

xy2 = zip(usplit2, wtime2)
xy_sorted2 = sorted(xy2, key = lambda pair: pair[0])
usplit_sorted2, wtime_sorted2 = zip(*xy_sorted2) 

wtime2 = np.array(wtime_sorted2)
usplit2 = np.array(usplit_sorted2)

print(usplit)
print(usplit2)

p = np.polyfit(usplit, wtime, 1)
p2 = np.polyfit(usplit2, wtime2, 1)


fig, ax = plt.subplots()

ax.scatter(usplit, wtime, marker="^", color = "red", label = "n = 5")
ax.scatter(usplit2, wtime2, marker="*", color = "blue", label = "n = 9")
ax.plot(usplit, p[0]*usplit + p[1], color = "black")
ax.plot(usplit2, p2[0]*usplit2 + p2[1], color = "black")



ax.set_xlabel("Splitting layers", fontsize = 12)
ax.set_ylabel("Wall time (s)", fontsize = 12)

ax.set_title(f"Target unitary layer scaling (q = 12)")
ax.legend(loc= "lower right", fontsize=12)

fig.savefig(f"{data_dir}/plots/n{nlayers}_q{nqubits}_usplit_scaling.png", dpi = 300)




    
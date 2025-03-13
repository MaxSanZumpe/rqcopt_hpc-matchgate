import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


nqubits = 16
ulayers = 601

script_dir = os.path.dirname(__file__)
file_list = glob.glob(f"{script_dir}/opt_out/spl_hubbard2d_suzuki2_n*_q16_u601_t0.25s_g1.50_opt_iter10.hdf5")


wtime = []
lay = []
threads = []
tasks = []


for file in file_list:
    with h5py.File(os.path.join(script_dir, file), "r") as f:

        lay.append(f.attrs["nlayers"])
        wtime.append(f.attrs["Walltime"])
        threads.append(f.attrs["NUM_THREADS"])
        tasks.append(f.attrs["NUM_TASKS"])


print("MPI BENCH:")
for i in range(len(lay)):
    print(f"Layers: {lay[i]} Tasks: {tasks[i]}, threads: {threads[i]}, wtime: {wtime[i]} s")


threads = np.array(threads)
wtime = np.array(wtime)


xy = zip(lay, wtime)
xy1_sorted = sorted(xy, key = lambda pair: pair[0])
lay_sorted, wtime_sorted = zip(*xy1_sorted) 
lay = np.array(lay_sorted)
wtime = np.array(wtime_sorted)

ex = 1
p = np.polyfit(lay[ex:], wtime[ex:]/60/10, 2)
p1 = np.polyfit(lay[ex:], wtime[ex:]/60/10, 1)

x_fit = np.linspace(7, 35)

fit = np.polyval(p, x_fit)
fit1 = np.polyval(p1, x_fit)


print(p)
print(p1)
s = f"{p1[0]:.2f}$n$+{np.abs(p1[1]):.1f}"
z = f"{p[0]:.2f}$n^2$-{np.abs(p[1]):.2f}$n$ + {np.abs(p[2]):.1f}"

fig, ax = plt.subplots()

wtime[0] = wtime[0]/2
ax.scatter(lay[ex:], wtime[ex:]/60/10, marker="^", color = "red", )
ax.plot(x_fit, fit, color="black", label=f"Fit: {z}")
#ax.plot(lay, fit1, color="black", label=f"Fit: {s}")

print(fit[0])

ax.set_title(f"Wall time benchmark (q = 16)", fontsize= 12)

ax.set_xlabel("Circuit layers", fontsize = 12)
ax.set_ylabel("Wall time (min)", fontsize = 12)

ax.xaxis.set_major_locator(FixedLocator(lay))


ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi_u{ulayers}_layers_scaling", dpi=300)


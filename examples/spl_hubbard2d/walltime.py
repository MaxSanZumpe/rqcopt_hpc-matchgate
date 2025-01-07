import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

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


print(lay)
print(wtime/60/60)

fig, ax = plt.subplots()

ax.scatter(lay[1:], wtime[1:]/60, marker=".", color = "black")


ax.set_title(f"Lay bech", fontsize= 12)

ax.set_xlabel("Circuit layers", fontsize = 12)
ax.set_ylabel("Wall time (min)", fontsize = 12)

#ax.legend(fontsize = 12)
fig.savefig(f"{script_dir}/mpi_u{ulayers}_layers_scaling", dpi=300)


import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

nqubits = 16
nlayers = 5
ulayers = 253

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
cores = threads*tasks
xy = zip(cores, wtime)
xy1_sorted = sorted(xy, key = lambda pair: pair[0])
cores_sorted, wtime_sorted = zip(*xy1_sorted) 
cores = np.array(cores_sorted)
wtime = np.array(wtime_sorted)


# fig, ax = plt.subplots()

# ax.scatter(c
# ax.scatter(cores , wtime2[0]/wtime, marker=".", color = "black", label = "MPI;  56 threads/task")
# ax.scatter(cores3, wtime2[0]/wtime3, marker=".", color = "blue", label = "MPI; 112 threads/task")
# ax.plot([56, 448], [1, 8], color = "green", label = "Ideal scaling")

# ax.set_title(f"MPI benchmark (q = {nqubits}; n = {nlayers})", fontsize= 12)

# ax.set_xlabel("Total threads", fontsize = 12)
# ax.set_ylabel("Wall time speed-up", fontsize = 12)

# ax.legend(fontsize = 12)
# fig.savefig(f"{script_dir}/mpi/plots/mpi_n{nlayers}_u{ulayers}", dpi=300)


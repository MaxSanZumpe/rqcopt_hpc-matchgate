import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nqubits = 16
nlayers = 5
ulayers = 601

script_dir = os.path.dirname(__file__)

file_list1 = glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_th112_010_*threads*.hdf5")
file_list2 = glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_th112_110_*threads*.hdf5")
file_list3 = glob.glob(f"{script_dir}/bench_out_mat4x4/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_th112_010_*threads*.hdf5")


if nqubits < 12:
    file_list1.append(glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_0*serial*.hdf5")[0])
    file_list2.append(glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_1*serial*.hdf5")[0])
    file_list3.append(glob.glob(f"{script_dir}/bench_out_mat4x4/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_0*serial*.hdf5")[0])

print(file_list1)
print(file_list2)
print(file_list3)

wtime1 = []
threads1 = []

wtime2 = []
threads2 = []

wtime3 = []
threads3 = []

for file1, file2, file3 in zip(file_list1, file_list2, file_list3):
    with h5py.File(file1, "r") as f:
        wtime1.append(f.attrs["Walltime"])
        threads1.append(f.attrs["NUM_THREADS"])

    with h5py.File(file2, "r") as f:
        wtime2.append(f.attrs["Walltime"])
        threads2.append(f.attrs["NUM_THREADS"])
    
    
    with h5py.File(file3, "r") as f:
        wtime3.append(f.attrs["Walltime"])
        threads3.append(f.attrs["NUM_THREADS"])

print(wtime2, threads2)
xy1 = zip(threads1, wtime1)
xy2 = zip(threads2, wtime2)
xy3 = zip(threads3, wtime3)

xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
xy2_sorted = sorted(xy2, key = lambda pair: pair[0])
xy3_sorted = sorted(xy3, key = lambda pair: pair[0])

threads1_sorted, wtime1_sorted = zip(*xy1_sorted) 
threads2_sorted, wtime2_sorted = zip(*xy2_sorted)
threads3_sorted, wtime3_sorted = zip(*xy3_sorted) 


wtime1 = np.array(wtime1_sorted)
wtime2 = np.array(wtime2_sorted)
wtime3 = np.array(wtime3_sorted)


threads1 = np.array(threads1_sorted)
threads2 = np.array(threads2_sorted)
threads3 = np.array(threads3_sorted)


mspu = (wtime3/wtime1)
ispu = (wtime3/wtime2)
tspu = (wtime1/wtime2)

print("Serial data:")
print(f"Threads                 : {threads1[0]}, {threads2[0]}, {threads3[0]}")
print(f"Matchgate speed-up      : {mspu[0]}")
print(f"Matchgate + INV speed-up: {ispu[0]}")
print(f"Walltime matchgates     : {wtime1[0]} s")
print(f"Walltime match +inv     : {wtime2[0]} s")
print(f"Walltime general        : {wtime3[0]} s")

print("112 Thread data:")
print(f"Threads                 : {threads1[-1]}, {threads2[-1]}, {threads3[-1]}")
print(f"Matchgate speed-up      : {mspu[-1]}")
print(f"Matchgate + INV speed-up: {ispu[-1]}")
print(f"Walltime matchgates     : {wtime1[-1]} s")
print(f"Walltime match +inv     : {wtime2[-1]} s")
print(f"Walltime general        : {wtime3[-1]} s")

temp = np.array([nlayers, mspu[0], ispu[0], wtime3[0], wtime1[0], wtime2[0], mspu[-1], ispu[-1], wtime3[-1], wtime1[-1], wtime2[-1]])

np.savetxt(f"{script_dir}/bench_out/q{nqubits}/plots/n{nlayers}_q{nqubits}_u{ulayers}_invariance_scaling.txt", temp)

fig, ax = plt.subplots()

ax.scatter(threads2, mspu, marker=".", label = "Matchgates", color = "black")
ax.scatter(threads3, ispu, marker="x", label = "Matchgates + Invariance", color = "green")

ax.set_xlabel("Thread number")
ax.set_ylabel("Walltime speed-up")

ax.set_title(f"Matchgate Benchmark: q = {nqubits}; n = {nlayers}")

ax.legend()
fig.savefig(f"{script_dir}/bench_out/q{nqubits}/plots/n{nlayers}_q{nqubits}_u{ulayers}_invariance_scaling.pdf")

import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


nqubits = 12
nlayers = 7
ulayers = 601

script_dir = os.path.dirname(__file__)


file_list1 = glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_th*_010*.hdf5")
file_list2 = glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_th*_110*.hdf5")
file_list3 = glob.glob(f"{script_dir}/bench_out_mat4x4/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_th*_010*.hdf5")


if nqubits != 16:
    file_list1.append(glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_0*serial*.hdf5")[0])
    file_list2.append(glob.glob(f"{script_dir}/bench_out/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_1*serial*.hdf5")[0])
    file_list3.append(glob.glob(f"{script_dir}/bench_out_mat4x4/q{nqubits}/n{nlayers}_q{nqubits}_u{ulayers}_0*serial*.hdf5")[0])


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


# mspu = 100*(wtime3 - wtime1)/wtime3
# ispu = 100*(wtime3 - wtime2)/wtime3
# tspu = 100*(wtime1 - wtime2)/wtime1


mspu = (wtime3/wtime1)
ispu = (wtime3/wtime2)
tspu = (wtime1/wtime2)

mean_mspu = np.mean(mspu)
mean_ispu = np.mean(ispu)
mean_tspu = np.mean(tspu)


print(f"Matchgate speed-up      : {mean_mspu}")
print(f"Matchgate + INV speed-up: {mean_ispu}")
print(f"Intermediate speed-up   : {mean_tspu}")

temp = np.array([mean_mspu, mean_ispu, mean_tspu])

np.savetxt(f"{script_dir}/bench_out/q{nqubits}/plots/n{nlayers}_q{nqubits}_u{ulayers}_invariance_scaling.txt", temp)

fig, ax = plt.subplots()

ax.scatter(threads2, mspu, marker="^", label = "Matchgates", color = "blue")
ax.scatter(threads3, ispu, marker="v", label = "Matchgates + Invariance", color = "green")
#ax.scatter(threads2, tspu, marker="x", label = "Intermediate", color = "green")



ax.set_xlabel("Thread number")
ax.set_ylabel("speed-up")

ax.set_title(f"Benchmark: qubits = {nqubits}; layers = {nlayers}")

ax.legend()
fig.savefig(f"{script_dir}/bench_out/q{nqubits}/plots/n{nlayers}_q{nqubits}_u{ulayers}_invariance_scaling.png")


    
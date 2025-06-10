import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

qlist = [8, 10, 12, 16]
ulayers = 601

script_dir = os.path.dirname(__file__)

arr = []

for q in qlist:
    file_list = glob.glob(f"{script_dir}/bench_out/q{q}/plots/n*_q{q}_u{ulayers}_invariance_scaling.txt")
    
    tn = []
    a = []
    b = []

    for file in file_list:
        content = np.loadtxt(file)
        tn.append(content[0])
        a.append(content[1])
        b.append(content[2])

    
    temp = zip(tn, a, b)
    temp_sorted = sorted(temp, key = lambda pair: pair[0])
    tn, a, b = zip(*temp_sorted) 

    arr.append([np.array(tn), np.array(a), np.array(b)])

cmap = ["black", "green", "red", "purple"]
mmap = [".", "x", "v", "*"]

fig, ax = plt.subplots()

for i in range(len(qlist)):
    ax.plot(arr[i][0], arr[i][1], color = cmap[i], marker= mmap[i], label = f"q = {qlist[i]}")

ax.tick_params(axis="x", labelsize = 12)
ax.tick_params(axis="y", labelsize = 12)

ax.set_title("Matchgate speed-ups map", fontsize = 18)
ax.set_xlabel("Circuit Layers", fontsize = 20)
ax.set_ylabel("Walltime speed-up", fontsize = 20)

ax.legend(fontsize = 16)
fig.savefig(f"{script_dir}/spu_match_map_u{ulayers}_.pdf", dpi = 400)


fig, ax = plt.subplots()

for i in range(len(qlist)):
    ax.plot(arr[i][0], arr[i][2], color = cmap[i], marker= mmap[i], label = f"q = {qlist[i]}")

ax.tick_params(axis="x", labelsize = 12)
ax.tick_params(axis="y", labelsize = 12)

ax.set_title("Matchgate + Invariance speed-ups", fontsize = 18)
ax.set_xlabel("Circuit Layers", fontsize = 20)
ax.set_ylabel("Walltime speed-up", fontsize = 20)
ax.legend(fontsize = 16)

fig.savefig(f"{script_dir}/spu_inv_map_u{ulayers}_.pdf", dpi=400)


    
import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

qlist = [8, 10, 12]
ulayers = 601

script_dir = os.path.dirname(__file__)

spu_arr = []

for q in qlist:
    file_list = glob.glob(f"{script_dir}/bench_out/q{q}/plots/n*_q{q}_u{ulayers}_invariance_scaling.txt")
    
    tn = []
    a = []
    b = []

    for file in file_list:
        content = np.loadtxt(file)
        tn.append(content[3])
        a.append(content[0])
        b.append(content[1])

    
    
    temp = zip(tn, a, b)
    temp_sorted = sorted(temp, key = lambda pair: pair[0])
    tn, a, b  = zip(*temp_sorted) 

    spu_arr.append(np.array([tn, a, b]))

        
cmap = ["green", "blue", "red"]
mmap = [".", "x", "v"]

fig, ax = plt.subplots()

for i in range(len(qlist)):
    ax.plot(spu_arr[i][0], spu_arr[i][1], color = cmap[i], marker= mmap[i], label = f"q = {qlist[i]}")


ax.set_title("Matchgate speed-up map")
ax.set_xlabel("Circuit Layers")
ax.set_ylabel("Walltime speed-up")

ax.legend()
fig.savefig(f"{script_dir}/spu_match_map_u{ulayers}_.png")


fig, ax = plt.subplots()

for i in range(len(qlist)):
    ax.plot(spu_arr[i][0], spu_arr[i][2], color = cmap[i], marker= mmap[i], label = f"q = {qlist[i]}")

ax.set_title("Matchgate + Invariance speed-up map")
ax.set_xlabel("Circuit Layers")
ax.set_ylabel("Walltime speed-up")
ax.legend()

fig.savefig(f"{script_dir}/spu_inv_map_u{ulayers}_.png")


    
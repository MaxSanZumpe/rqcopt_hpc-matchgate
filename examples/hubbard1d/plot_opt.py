import os
import h5py
import numpy as np
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator


L = 6
q = 2*L
g = 4.0
t = 1.0

opt_arr = []
ini_arr = []
layers_arr = []


script_dir = os.path.dirname(__file__)

file_list = glob.glob(f"{script_dir}/opt_out/q{q}/hubbard1d_*u1009*.hdf5")

print(file_list)

for file in file_list:
    with h5py.File(file, "r") as f:
        tmp = np.array(f["f_iter"])
        layers_arr.append([f.attrs["nlayers"]])
        ini_arr.append(2*tmp[0] + 2*2**q)
        opt_arr.append(2*tmp[-1] + 2*2**q)
        print(f"layers: {f.attrs['nlayers']}, Initial norm = {2*tmp[0] + 2*2**q} -> Final norm = {2*tmp[-1] + 2*2**q}")


fig, ax = plt.subplots()
ax.scatter(layers_arr, opt_arr, marker = "^", color = "green", label = "Optimized gates")
ax.scatter(layers_arr, ini_arr, marker = ".", color = "black", label = "Initial gates")


ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Layers")
ax.set_ylabel("$|| U - W(G) ||_F$")
#ax.xaxis.set_major_locator(FixedLocator([10, 100, 600]))


ax.set_title(f"Hubbard (2D): Qubits = 4x4, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, f"opt_out/q{q}/hubb2d_g{g:.2f}_t{t:.2f}_opt_norms.png"))
    
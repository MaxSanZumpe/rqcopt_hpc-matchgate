import os
import h5py
import numpy as np
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator


L = 8
q = 2*L
g = 1.5
t = 0.25

opt_arr = []
ini_arr = []
layers_arr = []


script_dir = os.path.dirname(__file__)

file_list = glob.glob(f"{script_dir}/opt_out/spl_hubbard2d_*u601*.hdf5")

for file in file_list:
    with h5py.File(file, "r") as f:
        tmp = np.array(f["f_iter"])
        layers_arr.append([f.attrs["nlayers"]])
        ini_arr.append((2*tmp[0] + 2*2**16)/2**q)
        opt_arr.append((2*tmp[-1] + 2*2**16)/2**q)


print(ini_arr)
print(opt_arr)

fig, ax = plt.subplots()
ax.scatter(layers_arr, opt_arr, marker = "^", color = "green", label = "Optimized gates")
ax.scatter(layers_arr, ini_arr, marker = ".", color = "black", label = "Initial gates")


ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Layers")
ax.set_ylabel("$\\rho_{error}$")
#ax.xaxis.set_major_locator(FixedLocator([10, 100, 600]))


ax.set_title(f"Hubbard (2D): Qubits = 4x4, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, f"opt_out/hubb2d_g{g:.2f}_t{t:.2f}_opt_norms.png"))
    
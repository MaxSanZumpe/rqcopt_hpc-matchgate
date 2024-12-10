import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

L = 8
q = 2*L
g = 4.0
t = 1


errors_arr = []
layers_arr = []

script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, f"spl_hubbard1d_suzuki2_q{q}_us{125}_u{251}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"spl_hubbard1d_suzuki4_q{q}_us{25}_u{251}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)


file_path = os.path.join(script_dir, f"spl_hubbard1d_blanes_moan4_q{q}_us{21}_u{253}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)


fig, ax = plt.subplots()
ax.scatter(layers_arr[0], errors_arr[0], marker = ".", color = "black", label = "Strang o4")
ax.scatter(layers_arr[1], errors_arr[1], marker = "o", color = "green", label = "Suzuki order 4")
ax.scatter(layers_arr[2], errors_arr[2], marker = "^", color = "red", label = "Blanes Moan o4")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Layers")
ax.set_ylabel("$|| \psi - \psi_0 ||^2$")

ax.set_title(f"Spinless Hubbard (1D):  J = {1}, U = {g}, t = {t}s")
ax.legend()
#ax.plot(dt, fit, color = "red")
fig.savefig(os.path.join(script_dir, "erros.png"))

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

q = 16
g = 4.0
t = 0.5


errors_arr = []
layers_arr = []

script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, f"spl_hubbard2d_suzuki2_q{q}_us{42}_u{253}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

# file_path = os.path.join(script_dir, f"hubbard2d_suzuki4_q{q}_us{13}_u{261}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
# with h5py.File(file_path, "r") as f:
#     errors = np.array(f["errors"])
#     layers = np.array(f["layers"])

# errors_arr.append(errors)
# layers_arr.append(layers)


# file_path = os.path.join(script_dir, f"hubbard1d_auzinger6_q{q}_us{9}_u{253}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
# with h5py.File(file_path, "r") as f:
#     errors = np.array(f["errors"])
#     layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)


p0 = np.polyfit(np.log(layers_arr[0]), np.log(errors_arr[0]), 1)
print(f"suzuki (order 2) slope: {p0[0]} layers^-1")

fig, ax = plt.subplots()
ax.scatter(layers_arr[0], errors_arr[0], marker = ".", color = "black", label = "strang")
ax.plot(layers_arr[0], np.exp(p0[0]*np.log(layers_arr[0])+p0[1]))
#ax.scatter(layers_arr[1], errors_arr[1], marker = "o", color = "green", label = "suzuki o4")
#ax.scatter(layers_arr[2], errors_arr[2], marker = "^", color = "red", label = "auzinger o6")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Layers")
ax.set_ylabel("$|| \psi - \psi_0 ||^2$")

ax.set_title(f"Hubbard (2D):  J = {1}, U = {g}, t = {t}s")
ax.legend()
#ax.plot(dt, fit, color = "red")
fig.savefig(os.path.join(script_dir, "erros.png"))

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


L = 8
q = 2*L
g = 4
t = 1


errors_arr = []
layers_arr = []

script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, f"hubbard1d_suzuki2_q{q}_us{75}_u{301}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"hubbard1d_suzuki4_q{q}_us{15}_u{301}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"hubbard1d_yoshida4_q{q}_us{25}_u{301}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)


file_path = os.path.join(script_dir, f"hubbard1d_auzinger6_q{q}_us{11}_u{309}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

p0 = np.polyfit(np.log(layers_arr[0]), np.log(errors_arr[0]), 1)
p1 = np.polyfit(np.log(layers_arr[1]), np.log(errors_arr[1]), 1)
p2 = np.polyfit(np.log(layers_arr[2]), np.log(errors_arr[2]), 1)
p3 = np.polyfit(np.log(layers_arr[3]), np.log(errors_arr[3]), 1)

print("Slopes:")
print(f"-Strang: {p0[0]} 1/layers")
print(f"-Suzuki: {p1[0]} 1/layers")
print(f"-Yoshid: {p2[0]} 1/layers")
print(f"-Auzing: {p3[0]} 1/layers")


fig, ax = plt.subplots()
ax.plot(layers_arr[0], errors_arr[0], marker = ".", color = "black", label = "Strang (order 2)")
ax.plot(layers_arr[1], errors_arr[1], marker = "v", color = "green", label = "Suzuki (order 4)")
ax.plot(layers_arr[2], errors_arr[2], marker = "^", color = "blue",  label = "Yoshida (order 4)")
ax.plot(layers_arr[3], errors_arr[3], marker = "+", color = "red",   label = "Auzinger (order 6)")

ax.set_xscale("log")
ax.set_yscale("log")
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Layers")
ax.set_ylabel("$|| \psi - \psi_0 ||^2$")

ax.set_title(f"Hubbard (1D): Qubits = {q} J = {1}, U = {g}, t = {t}s")
ax.legend()
#ax.plot(dt, fit, color = "red")
fig.savefig(os.path.join(script_dir, "erros.png"))

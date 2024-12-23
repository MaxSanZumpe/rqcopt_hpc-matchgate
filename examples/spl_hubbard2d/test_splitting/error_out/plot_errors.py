import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

q = 16
g = 1.5
t = 0.25


errors_arr = []
layers_arr = []

script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, f"spl_hubbard2d_suzuki2_q{q}_us{150}_u{901}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"spl_hubbard2d_suzuki4_q{q}_us{30}_u{901}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"spl_hubbard2d_suzuki6_q{q}_us{6}_u{901}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)



p0 = np.polyfit(np.log(layers_arr[0]), np.log(errors_arr[0]), 1)
p1 = np.polyfit(np.log(layers_arr[1]), np.log(errors_arr[1]), 1)
p2 = np.polyfit(np.log(layers_arr[2]), np.log(errors_arr[2]), 1)


print("Slopes:")
print(f"Strang (order 2) : {p0[0]} 1/layers")
print(f"Suzuki (order 4) : {p1[0]} 1/layers")
print(f"Suzuki (order 6) : {p2[0]} 1/layers")


fig, ax = plt.subplots()
ax.plot(layers_arr[0], errors_arr[0], marker = ".", color = "black", label = "Strang (order 2)")
ax.plot(layers_arr[1], errors_arr[1], marker = "v", color = "green", label = "Suzuki (order 4)")
ax.plot(layers_arr[2], errors_arr[2], marker = "^", color = "red"  , label = "Suzuki (order 6)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Layers")
ax.set_ylabel("$|| \psi_{approx} - \psi_{exact} ||_\infty$")

ax.set_title(f"Spinless Hubbard (2D): Qubits: 4x4, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, "splh2d_erros.png"))

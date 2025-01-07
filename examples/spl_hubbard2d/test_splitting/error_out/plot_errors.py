import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator


L = 8
q = 2*L
g = 1.5
t = 0.25

nlayers = 31

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
print(f"-Strang: {p0[0]} 1/layers")
print(f"-Suzuk4: {p1[0]} 1/layers")
print(f"-Suzuk6: {p2[0]} 1/layers")


fig, ax = plt.subplots()
ax.plot(layers_arr[0], errors_arr[0], marker = ".", color = "black" , label = "Strang (2)")
ax.plot(layers_arr[1], errors_arr[1], marker = "*", color = "purple", label = "Suzuki (4)")
ax.plot(layers_arr[2], errors_arr[2], marker = "x", color = "red"   , label = "Suzuki (6)")


ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Splitting Layers", fontsize =12)
ax.set_ylabel("$|| \psi_{approx.} - \psi_{exact} ||_{\infty}$", fontsize=12)
ax.xaxis.set_major_locator(FixedLocator([10, 100, 900]))


ax.set_title(f"Spinless Hubbard (2D): Qubits = {q}, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, f"hubb2d_q{q}_g{g:.2f}_t{t:.2f}_errors.png"), dpi=300)

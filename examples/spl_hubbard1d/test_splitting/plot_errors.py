import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator


q = 14
g = 4.0
t = 1


errors_arr = []
layers_arr = []

script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, f"error_out/spl_hubbard1d_suzuki2_q{q}_us{300}_u{601}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors[1:])
layers_arr.append(layers[1:])

file_path = os.path.join(script_dir, f"error_out/spl_hubbard1d_suzuki4_q{q}_us{60}_u{601}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"error_out/spl_hubbard1d_yoshida4_q{q}_us{100}_u{601}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)


file_path = os.path.join(script_dir, f"error_out/spl_hubbard1d_blanes4_q{q}_us{50}_u{601}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)


file_path = os.path.join(script_dir, f"error_out/spl_hubbard1d_suzuki6_q{q}_us{12}_u{601}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)


p0 = np.polyfit(np.log(layers_arr[0]     ), np.log(errors_arr[0]     ), 1)
p1 = np.polyfit(np.log(layers_arr[1][10:]), np.log(errors_arr[1][10:]), 1)
p2 = np.polyfit(np.log(layers_arr[2][11:]), np.log(errors_arr[2][11:]), 1)
p3 = np.polyfit(np.log(layers_arr[3]     ), np.log(errors_arr[3]     ), 1)
p4 = np.polyfit(np.log(layers_arr[4]     ), np.log(errors_arr[4]     ), 1)



print("Slopes:")
print(f"-Strang: {p0[0]} 1/layers")
print(f"-Suzuk4: {p1[0]} 1/layers")
print(f"-Yoshid: {p2[0]} 1/layers")
print(f"-Blanes: {p3[0]} 1/layers")
print(f"-Suzuk6: {p4[0]} 1/layers")



fig, ax = plt.subplots()
ax.plot(layers_arr[0], errors_arr[0], marker = ".", color = "black",  label = "Strang (order 2)")
ax.plot(layers_arr[1], errors_arr[1], marker = "v", color = "green",  label = "Suzuki (order 4)")
ax.plot(layers_arr[2], errors_arr[2], marker = "^", color = "blue",   label = "Yoshida (order 4)")
ax.plot(layers_arr[3], errors_arr[3], marker = "+", color = "purple", label = "Blanes Moan (order 4)")
ax.plot(layers_arr[4], errors_arr[4], marker = "+", color = "red",    label = "Suzuki (order 6)")



ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_ylabel("$|| \psi_{approx.} - \psi_{exact} ||_{\infty}$")
ax.xaxis.set_major_locator(FixedLocator([10, 100, 600]))

ax.set_title(f"Spinless Hubbard (1D): Qubits = {q}, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, f"plots/splh1d_g{g:.2f}_t{t:.2f}_errors.png"))

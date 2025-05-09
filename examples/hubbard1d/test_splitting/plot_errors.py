import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator


L = 6
q = 2*L
g = 4
t = 1.00

errors_arr = []
layers_arr = []

script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, f"error_out/hubbard1d_suzuki2_q{q}_us{200}_u{801}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"error_out/hubbard1d_suzuki4_q{q}_us{40}_u{801}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"error_out/hubbard1d_yoshida4_q{q}_us{66}_u{793}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"error_out/hubbard1d_auzinger6_q{q}_us{29}_u{813}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"error_out/hubbard1d_suzuki6_q{q}_us{8}_u{801}_t{t:.2f}s_g{g:.2f}_errors.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)



file_path = os.path.join(script_dir, f"error_out/hubbard1d_suzuki2_opt_n{33}_q{q}_t{t:.2f}s_g{g:.2f}_iter15.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

file_path = os.path.join(script_dir, f"error_out/hubbard1d_suzuki4_opt_n{41}_q{q}_t{t:.2f}s_g{g:.2f}_iter15.hdf5")
with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])
    layers = np.array(f["layers"])

errors_arr.append(errors)
layers_arr.append(layers)

p0 = np.polyfit(np.log(layers_arr[0][10:]), np.log(errors_arr[0][10:]), 1)
p1 = np.polyfit(np.log(layers_arr[1][12:]), np.log(errors_arr[1][12:]), 1)
p2 = np.polyfit(np.log(layers_arr[2][20:]), np.log(errors_arr[2][20:]), 1)
p3 = np.polyfit(np.log(layers_arr[3][12:]), np.log(errors_arr[3][12:]), 1)
p4 = np.polyfit(np.log(layers_arr[4][1:]), np.log(errors_arr[4][1:]), 1)

print("Slopes:")
print(f"-Strang: {p0[0]} 1/layers")
print(f"-Suzuki: {p1[0]} 1/layers")
print(f"-Yoshid: {p2[0]} 1/layers")
print(f"-Auzing: {p3[0]} 1/layers")
print(f"-Suzuk6: {p4[0]} 1/layers")


fig, ax = plt.subplots()
ax.plot(layers_arr[0], errors_arr[0], marker = ".", color = "black", label = "Strang (order 2)")
ax.plot(layers_arr[1], errors_arr[1], marker = "v", color = "purple", label = "Suzuki (order 4)")
ax.plot(layers_arr[2], errors_arr[2], marker = "x", color = "blue",  label = "Yoshida (order 4)")
ax.plot(layers_arr[3], errors_arr[3], marker = "+", color = "red",   label = "Auzinger (order 6)")
#ax.plot(layers_arr[4], errors_arr[4], marker = "+", color = "green",   label = "Suzuki (order 6)")


ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Layers")
ax.set_ylabel("$|| \psi_{approx.} - \psi_{exact} ||_{\infty}$")
#ax.xaxis.set_major_locator(FixedLocator([10, 100, 600]))


ax.set_title(f"Hubbard (1D): Qubits = {q}, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, f"plots/hubb1d_q{q}_g{g:.2f}_t{t:.2f}_uerrors.png"))


extent = [10, 4, 8, 4]

fig, ax = plt.subplots()
ax.plot(layers_arr[0][:extent[0]], errors_arr[0][:extent[0]], marker = ".", color = "black", label = "Strang (order 2)")
ax.plot(layers_arr[1][:extent[1]], errors_arr[1][:extent[1]], marker = "v", color = "purple", label = "Suzuki (order 4)")
ax.plot(layers_arr[2][:extent[2]], errors_arr[2][:extent[2]], marker = "x", color = "blue",  label = "Yoshida (order 4)")
ax.plot(layers_arr[3][:extent[3]], errors_arr[3][:extent[3]], marker = "+", color = "red",   label = "Auzinger (order 6)")

ax.plot(layers_arr[5], errors_arr[5], marker = "^", color = "green", label = "Optmized gates")
ax.plot(layers_arr[6], errors_arr[6], marker = "^", color = "green",)



ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Layers")
ax.set_ylabel("$|| \psi_{approx.} - \psi_{exact} ||_{\infty}$")
#ax.xaxis.set_major_locator(FixedLocator([10, 100, 600]))


ax.set_title(f"Hubbard (1D): Qubits = {q}, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, f"plots/hubb1d_q{q}_g{g:.2f}_t{t:.2f}_opterrors.png"))
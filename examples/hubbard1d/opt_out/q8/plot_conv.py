import os
import h5py
import numpy as np
from matplotlib import pyplot as plt



L = 4
q = 2*L
g = 4.0
t = 0.2

ulayers = 0

script_dir = os.path.dirname(__file__)

file = f"{script_dir}/hubbard1d_suzuki4_n21_q{q}_u0_t{t:.2f}s_g{g:.2f}_conv200_inv0_opt.hdf5)"
file = "/home/max/rqcopt_hpc-matchgate/examples/hubbard1d/opt_out/q8/hubbard1d_suzuki4_n21_q8_u0_t0.20s_g4.00_conv200_inv0_opt.hdf5"

f_citer = []
iter = []


with h5py.File(file, "r") as f:
    f_citer.append(np.array(f["f_citer"]))

f_citer = np.array(f_citer[0])
print(f_citer.shape)

frob = (2*2**q+2*f_citer[:, 1])/(2*2**q)

print(frob.shape)

fig, ax = plt.subplots()
ax.plot(frob)
ax.set_xlabel("Iterations", fontsize = 12)

ax.set_title(f"Hubbard (1D): Qubits = {q}, J = {1}, U = {g}, t = {t}s")

fig.savefig("convergence.png", dpi = 300)


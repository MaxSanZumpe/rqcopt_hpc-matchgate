import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

L = 8
q = 2*L
g = 4.0

us = 40
ulayers = 161

t = 1

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, f"error_data/hubbard1d_suzuki2_q{q}_us{us}_u{ulayers}_t{t:.2f}s_g{g:.2f}_errors.hdf5")

with h5py.File(file_path, "r") as f:
    errors = np.array(f["errors"])

dt = np.array([t/s for s in range(1, us+1)])

p = np.polyfit(np.log(dt), np.log(errors), 1)

fit = p[0]*dt + p[1]

print(p[0])

fig, ax = plt.subplots()
ax.scatter(dt, errors, marker = "x",color = "black")
ax.set_yscale("log")
ax.set_xscale("log")
#ax.plot(dt, fit, color = "red")
fig.savefig(os.path.join(script_dir, "erros.png"))

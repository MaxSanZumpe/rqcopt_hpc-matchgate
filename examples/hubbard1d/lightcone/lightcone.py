import os
import re
import h5py
import numpy as np
from matplotlib import pyplot as plt


L = 8
q = 2*L
g = 4.0
ulayers = 561

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, f"hubbard1d_q{q}_g{g:.2f}_u{ulayers}_lightcone.hdf5")

with h5py.File(file_path, "r") as f:

    q = f.attrs["nqubits"]
    delta = f.attrs["delta"]
    data = []
    t_step = []

    # Sort by the numerical part
    keys = sorted(f.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))

    for key in keys:
        a = re.findall(r'\d+', key)[0]
        t_step.append(int(a))
        data.append(np.array(f[key][:, 0]))

data = np.array(data)
t_step = np.array(t_step)
t = t_step*delta


script_dir = os.path.dirname(__file__)
plt.imshow(np.roll(data, shift=(L-1)//2, axis=1)[:,:7], 
           interpolation="nearest", aspect="auto", origin="lower",
           extent=(-L//2+0.5, L//2-0.5, 0, delta*len(t_step)))

vel = 1.5
T = (L//2) / vel
plt.colorbar()
plt.plot([+0.5,  0.5 + T*vel], [ 0, T], "w")
plt.plot([-0.5, -0.5 - T*vel], [ 0, T], "w")

plt.xlabel("$\Delta j$")
plt.ylabel("t (s)")
plt.title(fr"$\langle \psi | n_{{j\uparrow}}(t) n_{{1\uparrow}}(0) | \psi \rangle$ for $H_{{hubb}}, U = 4$; velocity: {vel} $s^{{-1}}$")
plt.savefig(os.path.join(script_dir, f"plots/hubb1d_q{q}_g{g:.2f}_u{ulayers}_lightcone_c.png"))




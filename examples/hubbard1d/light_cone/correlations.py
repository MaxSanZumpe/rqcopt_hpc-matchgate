import os
import re
import h5py
import numpy as np
from matplotlib import pyplot as plt

L = 8
q = 2*L
g = 4.0
ulayers = 161

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, f"hubbard1d_q{q}_g{g:.2f}_u{ulayers}_correlations.hdf5")

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
diff = np.array(list(range(int(q/2))))

print(data.shape)

#fig, ax = plt.subplots()
#ax.pcolormesh(diff, t, data, cmap = "plasma", shading = "auto")
#fig.savefig(os.path.join(script_dir, "lightcone.png"))

script_dir = os.path.dirname(__file__)
vel = 1.25
plt.imshow(np.roll(data, shift=(L-1)//2, axis=1).real, 
           interpolation="nearest", aspect="auto", origin="lower",
           extent=(-L//2+0.5, L//2+0.5, 0, delta*len(t)))

plt.colorbar()
plt.plot([ 0.5,  1+L//2], [0, L//2*1/vel], "w")
plt.plot([-0.5, -L//2], [0, L//2*1/vel], "w")
plt.savefig(os.path.join(script_dir, "lightcone_c.png"))


import os
import re
import h5py
import numpy as np
from matplotlib import pyplot as plt

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir,"spl_hubbard1d_g4.0_correlations.hdf5")

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
diff = np.array(list(range(q)))

print(diff.shape)
print(t.shape)

fig, ax = plt.subplots()
ax.pcolormesh(diff, t, data, cmap = "plasma", shading = "auto")
fig.savefig(os.path.join(script_dir, "correlations.png"))


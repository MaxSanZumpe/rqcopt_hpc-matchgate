import os
import numpy as np
import matplotlib.pyplot as plt

file_dir  = os.path.dirname(__file__)
file_path = os.path.join(file_dir, "corr_data")

corr = np.loadtxt(file_path)

steps = 100
corr_x = corr[:steps, 0:4]
print(corr_x.shape)

L = 4

vel = 1.5


plt.imshow(np.roll(corr_x, shift=(L-1)//2, axis=1).real,
            interpolation="nearest", aspect="auto",
            origin="lower", extent=(-L//2+0.5, L//2+0.5, 0, 0.02*steps))
plt.xlabel("$\Delta j$")
plt.ylabel("t")
plt.title(fr"$\langle \psi | n_j(t) n_1(0) | \psi \rangle$ for $H_{{splh}}, U = 4;$ vel: {vel} $s^{{-1}}$")
plt.colorbar()
#plt.plot([ 0.5,  1+L//2], [0, L//2*1/vel], "w")
#plt.plot([-0.5, -L//2], [0, L//2*1/vel], "w")
plt.savefig(f"{file_dir}/splh2d_lightcone_x.png")


# for step in range(0, 100, 2):

#     fig, ax = plt.subplots()
#     ax.imshow(np.roll(np.roll(corr[step], shift = shifty, axis = 0 ), shift=shiftx, axis = 1), interpolation="nearest", aspect="auto", extent=[-0.5, 4 - 0.5, 4 - 0.5, -0.5])
#     fig.savefig(f"{file_dir}/lightcone2d_x.png")
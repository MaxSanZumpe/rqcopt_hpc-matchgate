import os
import numpy as np
import matplotlib.pyplot as plt

file_dir  = os.path.dirname(__file__)
file_path = os.path.join(file_dir, "corr_data")

corr = np.loadtxt(file_path)

corr = np.reshape(corr, (corr.shape[0], 4, 4))


shiftx = 1
shifty = 2

for step in range(0, 100, 2):

    fig, ax = plt.subplots()
    ax.imshow(np.roll(np.roll(corr[step], shift = shifty, axis = 0 ), shift=shiftx, axis = 1), interpolation="nearest", aspect="auto", extent=[-0.5, 4 - 0.5, 4 - 0.5, -0.5])
    fig.savefig(f"{file_dir}/testplots/test2d{step}.png")
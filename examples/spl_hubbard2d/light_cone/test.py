import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator

g = 1.5

file_dir  = os.path.dirname(__file__)
file_path = os.path.join(file_dir, "corr_data")

corr = np.loadtxt(file_path)

corr = np.reshape(corr, (corr.shape[0], 4, 4))


shiftx = 2
shifty = 1
for step in range(0, 100, 1):

    fig, ax = plt.subplots()
    im = ax.imshow(np.roll(np.roll(corr[step], shift = shifty, axis = 0), shift=shiftx, axis = 1), interpolation="nearest", aspect="auto", extent=[0.5, 4.5, 4.5, 0.5])
    
    fig.colorbar(im, ax=ax, orientation='vertical')

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4]))
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4]))

    ax.set_xlabel("i")
    ax.set_ylabel("j")

    ax.set_title(fr"$\langle \psi | n_{{ij}}(t = {step*0.005:.2f}s) n_{{2,3}}(0) | \psi \rangle$ for $H_{{splh}} (2D), U = 1.5$;")

    fig.savefig(f"{file_dir}/testplots/splh2d_g{g:.2f}_t{step*0.005:.3f}.png")
    plt.close()
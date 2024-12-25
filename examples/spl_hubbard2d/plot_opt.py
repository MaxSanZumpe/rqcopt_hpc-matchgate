import h5py
import os
import numpy as np

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "opt_out" ,f"spl_hubbard2d_suzuki2_n7_q16_u301_t0.25s_g1.50_opt_iter10.hdf5")
with h5py.File(file_path, "r") as f:
    print(f.keys())
    print(np.array(f["f_iter"]))

    for key in f.attrs.keys():
        print(f"{key} = {f.attrs[key]}")
    
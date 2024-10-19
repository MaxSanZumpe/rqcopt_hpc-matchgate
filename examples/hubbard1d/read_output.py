import h5py
import os

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "data" ,f"hubbard1d_opt_n{4}_q{14}_th56_110.hdf5")

with h5py.File(file_path, "r") as f:
    print(f.keys())
    print(f["vlist_opt"].shape)

    for key in f.attrs.keys():
        print(f"{key} = {f.attrs[key]}")
    
import h5py
import os

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "output_data" ,f"hubbard2d_trotter_order1_u20_n8_q4x4.hdf5")

with h5py.File(file_path, "r") as f:
    print(f.keys())
    print(f["vlist_opt"].shape)

    for key in f.attrs.keys():
        print(f"{key} = {f.attrs[key]}")
    
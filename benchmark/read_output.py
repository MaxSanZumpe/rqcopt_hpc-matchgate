import h5py
import os

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "output_data" ,"mpi_bench_n4_q16_tasks4_th28_01.hdf5")

with h5py.File(file_path, "r") as f:
    print(f.keys())
    

    for key in f.attrs.keys():
        print(f"{key} = {f.attrs[key]}")
    
import os
import h5py

q = 8

n = 5
spliting = "suzuki2"
u = 0

t = 1.00
g = 2.00

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, f"opt_out/q{q}" ,f"hubbard1d_{spliting}_n{n}_q{q}_u{u}_t{t:.2f}s_g{g:.2f}_iter1_opt.hdf5")

with h5py.File(file_path, "r") as f:
    print(f.keys())
    
    for key in f.attrs.keys():
        print(f"{key} = {f.attrs[key]}")
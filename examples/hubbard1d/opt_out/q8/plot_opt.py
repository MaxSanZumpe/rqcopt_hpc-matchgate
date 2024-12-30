import os
import h5py
import numpy as np
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator


L = 4
q = 2*L
g = 1.5
t = 0.25

ulayers = 0

script_dir = os.path.dirname(__file__)

file_list1 = glob.glob(f"{script_dir}/hubbard1d_suzuki2*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter10_inv0*.hdf5")
file_list2 = glob.glob(f"{script_dir}/hubbard1d_suzuki2*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter30_inv0*.hdf5")

file_list3 = glob.glob(f"{script_dir}/hubbard1d_suzuki4*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter10_inv0*.hdf5")
file_list4 = glob.glob(f"{script_dir}/hubbard1d_suzuki4*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter30_inv0*.hdf5")

file_list5 = glob.glob(f"{script_dir}/hubbard1d_suzuki4*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter30_inv1*.hdf5")

file_list6 = glob.glob(f"{script_dir}/hubbard1d_suzuki2*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*norm*.hdf5")
file_list7 = glob.glob(f"{script_dir}/hubbard1d_suzuki4*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*norm*.hdf5")
file_list8 = glob.glob(f"{script_dir}/hubbard1d_auzinger6*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*norm*.hdf5")

file_list9 = glob.glob(f"{script_dir}/hubbard1d_auzinger6*q{q}*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter30_inv0*.hdf5")


file_arr = [file_list1, file_list2, file_list3, file_list4, file_list5, file_list6, file_list7, file_list8, file_list9]

opt_arr = []
ini_arr = []
layers_arr = []

for file_list in file_arr:
    tmp_opt = []
    tmp_ini = []
    tmp_lay = []

    for file in file_list:
        with h5py.File(file, "r") as f:
            tmp = np.array(f["f_iter"])
            tmp_lay.append([f.attrs["nlayers"]])
            tmp_ini.append(np.sqrt((2*2**q+2*tmp[0])/2**q))
            tmp_opt.append(np.sqrt((2*2**q+2*tmp[-1])/2**q))
    
    
    xy1 = zip(tmp_lay, tmp_ini)
    xy2 = zip(tmp_lay, tmp_opt)
    xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
    xy2_sorted = sorted(xy2, key = lambda pair: pair[0])
    tmp_lay, tmp_ini = zip(*xy1_sorted)     
    tmp_lay, tmp_opt = zip(*xy2_sorted) 

    layers_arr.append(np.array(tmp_lay))
    ini_arr.append(np.array(tmp_ini))
    opt_arr.append(np.array(tmp_opt))


optt_lay = np.append(np.append(layers_arr[1][:4], layers_arr[3]), layers_arr[8])
optt_err = np.append(np.append(opt_arr[1][:4], opt_arr[3]), opt_arr[8])


xy1 = zip(optt_lay, optt_err)
xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
optt_lay, optt_err = zip(*xy1_sorted)      

fig, ax = plt.subplots()
ax.plot(np.append(layers_arr[0], layers_arr[5]), np.append(ini_arr[0],ini_arr[5]), 
        marker = ".", color = "black", label = "Suzuki 2")

ax.plot(np.append(layers_arr[2], layers_arr[6]), np.append(ini_arr[2],ini_arr[6]), 
        marker = "*", color = "purple", label = "Suzuki 4")

ax.plot(layers_arr[7], ini_arr[7], 
        marker = "x", color = "red", label = "Auzinger 6")

# ax.plot(np.append(layers_arr[0][:4], layers_arr[2]), np.append(opt_arr[0][:4],opt_arr[2]), 
#         marker = "^", color = "red", label = "Optimized gates; 15 iter")

ax.plot(optt_lay, optt_err, marker = "^", color = "green", label = "Optimized gates; 30 iter")


# ax.plot(layers_arr[8], opt_arr[8], 
#         marker = "^", color = "green", label = "Optimized gates; 30 iter")

# ax.plot(layers_arr[4], opt_arr[4], 
#        marker = "x", color = "blue", label = "INV 30 iter")



ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Layers")
ax.set_ylabel("$error$")
#ax.xaxis.set_major_locator(FixedLocator([10, 100, 600]))


ax.set_title(f"Hubbard (1D): Qubits = {q}, J = {1}, U = {g}, t = {t}s")
ax.legend()
fig.savefig(os.path.join(script_dir, f"hubb2d_g{g:.2f}_t{t:.2f}_opt_norms.png"))
    
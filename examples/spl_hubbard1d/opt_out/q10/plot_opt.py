import os
import h5py
import numpy as np
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator


q = 10
g = 1.5
t = 1

ulayers = 0

script_dir = os.path.dirname(__file__)

file_list1 = glob.glob(f"{script_dir}/spl_hubbard1d_suzuki2*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter30_inv1*.hdf5")
file_list2 = glob.glob(f"{script_dir}/spl_hubbard1d_blanes4*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter30_inv1*.hdf5")
file_list3 = glob.glob(f"{script_dir}/spl_hubbard1d_suzuki4*u{ulayers}_t{t:.2f}s_g{g:.2f}*_iter30_inv1*.hdf5")
file_list4 = glob.glob(f"{script_dir}/spl_hubbard1d_suzuki6*u{ulayers}_t{t:.2f}s_g{g:.2f}_norm.hdf5")


file_list5 = glob.glob(f"{script_dir}/spl_hubbard1d_suzuki2*u{ulayers}_t{t:.2f}s_g{g:.2f}_norm.hdf5")
file_list6 = glob.glob(f"{script_dir}/spl_hubbard1d_suzuki4*u{ulayers}_t{t:.2f}s_g{g:.2f}_norm.hdf5")
file_list7 = glob.glob(f"{script_dir}/spl_hubbard1d_blanes4*u{ulayers}_t{t:.2f}s_g{g:.2f}_norm.hdf5")


file_arr = [file_list1, file_list2, file_list3, file_list4, file_list5, file_list6, file_list7]

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


ex1 = 4
ex2 = 3

optt_lay = np.append(np.append(layers_arr[0][:ex1], layers_arr[1]), layers_arr[2][:ex2])
optt_err = np.append(np.append(opt_arr[0][:ex1], opt_arr[1]), opt_arr[2][:ex2])

xy1 = zip(optt_lay, optt_err)
xy1_sorted = sorted(xy1, key = lambda pair: pair[0])
optt_lay, optt_err = zip(*xy1_sorted)  

print(optt_lay[-1], layers_arr[6][4], layers_arr[3][-1])

fig, ax = plt.subplots()
ax.plot(np.append(layers_arr[0],layers_arr[4][:5]), np.append(ini_arr[0],ini_arr[4][:5]),
         marker = ".", color = "black", label = "Strang (2)")
ax.plot(np.append(layers_arr[1],layers_arr[6][:5]), np.append(ini_arr[1],ini_arr[6][:5]),
         marker = "*", color = "purple", label = "Blanes (4)")
ax.plot(np.append(layers_arr[2],layers_arr[5]), np.append(ini_arr[2],ini_arr[5]),
        marker = "x", color = "red", label = "Suzuki (4)")

ax.plot(layers_arr[3], ini_arr[3], marker = "s", color = "orange", label = "Suzuki (6)")

ax.plot(optt_lay, optt_err, marker = "^", color = "green", label = "Optmized gates")

print(opt_arr[1])
print(ini_arr[1])

# ax.plot(np.append(layers_arr[0][:], layers_arr[1]), np.append(opt_arr[0][:],opt_arr[1]), 
#         marker = "^", color = "red", label = "Optimized gates; 15 iter")

# ax.plot(np.append(layers_arr[1][:4], layers_arr[3]), np.append(opt_arr[0][:4],opt_arr[3]), 
#         marker = "^", color = "green", label = "Optimized gates; 30 iter")

# ax.plot(layers_arr[4], opt_arr[4], 
#         marker = "x", color = "blue", label = "INV 30 iter")


ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.set_xlabel("Circuit layers", fontsize = 12)
ax.set_ylabel("$\\rho_{error}$", fontsize = 12)
#ax.xaxis.set_major_locator(FixedLocator([10, 100, 600]))


ax.set_title(f"Spinless Hubbard (1D): Qubits = {q}, J = {1}, U = {g}, t = {t}s")
ax.legend(fontsize = 9)
fig.savefig(os.path.join(script_dir, f"spl_hubbard_g{g:.2f}_t{t:.2f}_opt_norms.png"), dpi = 300)

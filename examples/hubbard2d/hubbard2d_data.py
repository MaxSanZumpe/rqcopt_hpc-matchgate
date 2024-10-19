import os
import h5py
import numpy as np
import io_util as io
from splitting import spinless_hubbard_2d_trotter_order1_init, spinless_hubbard_2d_trotter_order2_init



def main():

    Lx = 4
    Ly = 4
    qdim = int(Lx*Ly)
    s = 2  #number of splitting steps

    # parameters for hamiltonian
    J = 1
    U = 0.75
    t = 0.25

    order = "order2"

    if (order == "order1"):
        Vblock_list_start, perms , nlayers = spinless_hubbard_2d_trotter_order1_init(J, U, t, s , Lx, Ly)
        Ublock_list      , uperms, ulayers = spinless_hubbard_2d_trotter_order1_init(J, U, t, 20, Lx, Ly)

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "input_data" ,f"hubbard2d_trotter_order1_u{ulayers}_n{nlayers}_q{Lx}x{Ly}_init.hdf5")

    if (order == "order2"):
        Vblock_list_start, perms , nlayers = spinless_hubbard_2d_trotter_order2_init(J, U, t, s , Lx, Ly)
        Ublock_list      , uperms, ulayers = spinless_hubbard_2d_trotter_order2_init(J, U, t, 15, Lx, Ly)

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "input_data" ,f"hubbard2d_trotter_order2_u{ulayers}_n{nlayers}_q{Lx}x{Ly}_init.hdf5")


    # save initial data to disk
    with h5py.File(file_path, "w") as file:

        file["nqubits"] = qdim

        file["nlayers"] = nlayers

        file["ulayers"] = ulayers

        file[f"Ulist"] = io.interleave_complex(np.stack(Ublock_list), "cplx")

        for i in range(ulayers):
            file[f"uperm{i}"] = np.arange(qdim) if uperms[i] is None else uperms[i]

        file[f"Vlist_start"] = io.interleave_complex(np.stack(Vblock_list_start), "cplx")

        for i in range(nlayers):
            print(perms[i])
            file[f"perm{i}"] = np.arange(qdim) if perms[i] is None else perms[i]
        # store parameters
        file.attrs["L"] = qdim
        file.attrs["J"] = float(J)
        file.attrs["U"] = float(U)
        file.attrs["t"] = float(t)

if __name__ == "__main__":
    main()
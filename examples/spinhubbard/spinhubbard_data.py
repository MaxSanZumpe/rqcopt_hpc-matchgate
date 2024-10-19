import os
import h5py
import numpy as np
import io_util as io
from splitting import spin_hubbart_strang_trotter_init



def main():

    s_list = list(range(1, 11))  #number of splitting steps
    qdim = 8

    for s in s_list:
        # parameters for hamiltonian
        J = 1
        U = 0.75
        t = 0.25
        
        #expiH = spinless_hubbard_unitary(qdim, J, U, t, "periodic")
        Vblock_list_start, perms , nlayers = spin_hubbart_strang_trotter_init(J, U, t, s , qdim, "matchgate")
        Ublock_list      , uperms, ulayers = spin_hubbart_strang_trotter_init(J, U, t, 20, qdim, "matchgate")

        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "data" ,f"spinhubbard_n{nlayers}_q{qdim}_init.hdf5")

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
                file[f"perm{i}"] = np.arange(qdim) if perms[i] is None else perms[i]
            # store parameters
            file.attrs["L"] = qdim
            file.attrs["J"] = float(J)
            file.attrs["U"] = float(U)
            file.attrs["t"] = float(t)

if __name__ == "__main__":
    main()
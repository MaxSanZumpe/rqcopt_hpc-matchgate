import os
import h5py
import numpy as np
import io_util as io
from splitting import spinless_hubbard_trotter_init



def main():

    s = 4  #number of splitting steps
    qdim = 14

    # parameters for hamiltonian
    J = 1
    U = 0.75
    t = 0.25
    
    #expiH = spinless_hubbard_unitary(qdim, J, U, t, "periodic")
    Vblock_list_start, perms , nlayers = spinless_hubbard_trotter_init(J, U, t, s , qdim, "matchgate")
    Ublock_list      , uperms, ulayers = spinless_hubbard_trotter_init(J, U, t, 10, qdim, "matchgate")

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "input_data" ,f"spinless_hubbard_n{nlayers}_q{qdim}_matchgate_init.hdf5")

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
        file.attrs["J"] = float(J)
        file.attrs["U"] = float(U)
        file.attrs["t"] = float(t)

if __name__ == "__main__":
    main()
import numpy as np
import h5py
import os
import rqcopt_matfree as oc
from io_util import interleave_complex


def apply_matchgate_data():

    rng = np.random.default_rng(52)

    L = 9

    A = oc.crandn((2, 2), rng)
    B = oc.crandn((2, 2), rng)
    V = oc.matchgate_matrix(A, B)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_apply_matchgate_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        file["v/center"] = interleave_complex(A, "cplx")
        file["v/corner"] = interleave_complex(B, "cplx")

        # random input statevector
        psi = oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, "cplx")

        # general i < j
        chi1 = oc.apply_gate(V, L, 2, 5, psi)
        # j < i
        chi2 = oc.apply_gate(V, L, 4, 1, psi)
        # j == i + 1
        chi3 = oc.apply_gate(V, L, 3, 4, psi)

        file["chi1"] = interleave_complex(chi1, "cplx")
        file["chi2"] = interleave_complex(chi2, "cplx")
        file["chi3"] = interleave_complex(chi3, "cplx")

        file.close()


def apply_matchgate_backward_data():

    # random number generator
    rng = np.random.default_rng(43)

    # system size
    L = 9

    A = oc.crandn((2, 2), rng)
    B = oc.crandn((2, 2), rng)
    V = oc.matchgate_matrix(A, B)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_apply_matchgate_backward_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        file["v/center"] = interleave_complex(A, "cplx")
        file["v/corner"] = interleave_complex(B, "cplx")

        # random input statevector
        psi = oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, "cplx")

        # fictitious upstream derivatives
        dpsi_out = oc.crandn(2**L, rng)
        dpsi_out /= np.linalg.norm(dpsi_out)
        file["dpsi_out"] = interleave_complex(dpsi_out, "cplx")


def apply_matchgate_backward_array_data():

    # random number generator
    rng = np.random.default_rng(76)

    # system size
    L = 9

    A = 0.5*oc.crandn((2, 2), rng)
    B = 0.5*oc.crandn((2, 2), rng)
    V = oc.matchgate_matrix(A, B)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_apply_matchgate_backward_array_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        file["v/center"] = interleave_complex(A, "cplx")
        file["v/corner"] = interleave_complex(B, "cplx")

        # random input statevector
        psi_list = [oc.crandn(2**L, rng) for _ in range(8)]
        psi_list = [psi/np.linalg.norm(psi) for psi in psi_list]
        psi_array = []
        for i in range(8):
            psi = psi_list[i]
            psi_array.append(psi)
            file[f"psi{i}"] = interleave_complex(psi, "cplx")
        
        # fictitious upstream derivatives
        dpsi_out = oc.crandn(2**L, rng)
        dpsi_out /= np.linalg.norm(dpsi_out)
        file["dpsi_out"] = interleave_complex(dpsi_out, "cplx")

        file["psi_array"]  = interleave_complex(np.asarray(psi_array).T, "cplx")


def apply_matchgate_to_array_data():

    # random number generator
    rng = np.random.default_rng(189)

    # system size
    L = 6
    # number of states
    nstates = 5

    A = 0.5*oc.crandn((2, 2), rng)
    B = 9.5*oc.crandn((2, 2), rng)
    V = oc.matchgate_matrix(A, B)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_apply_matchgate_to_array_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        file["v/center"] = interleave_complex(A, "cplx")
        file["v/corner"] = interleave_complex(B, "cplx")

        # random input statevectors
        psi_list = []
        chi1_list = []
        chi2_list = []
        chi3_list = []
        for _ in range(nstates):
            psi = oc.crandn(2**L, rng)
            psi /= np.linalg.norm(psi)
            psi_list.append(psi)

            # general i < j
            chi1 = oc.apply_gate(V, L, 2, 5, psi)
            # j < i
            chi2 = oc.apply_gate(V, L, 4, 1, psi)
            # j == i + 1
            chi3 = oc.apply_gate(V, L, 3, 4, psi)

            chi1_list.append(chi1)
            chi2_list.append(chi2)
            chi3_list.append(chi3)

        file["psi"]  = interleave_complex(np.asarray(psi_list).T, "cplx")
        file["chi1"] = interleave_complex(np.asarray(chi1_list).T, "cplx")
        file["chi2"] = interleave_complex(np.asarray(chi2_list).T, "cplx")
        file["chi3"] = interleave_complex(np.asarray(chi3_list).T, "cplx")


def apply_matchgate_placeholder_data():

    # random number generator
    rng = np.random.default_rng(56)

    # system size
    L = 7

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_apply_matchgate_placeholder_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        # random input statevector
        psi = oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, "cplx")

        # i < j
        i = 2
        j = 5
        # output statevector array
        psi_out1 = np.kron(psi, np.identity(4).reshape(-1))
        psi_out1 = psi_out1.reshape((2**i, 2, 2**(j - i - 1), 2, 2**(L - 1 - j), 2, 2, 2, 2))
        psi_out1 = psi_out1.transpose((0, 5, 2, 6, 4, 7, 8, 1, 3))
        psi_out1 = psi_out1.reshape(-1)

        file["psi_out1"] = interleave_complex(psi_out1, "cplx")

        # i > j
        i = 5
        j = 1
        # output statevector array
        psi_out2 = np.kron(psi, np.identity(4).reshape(-1))
        psi_out2 = psi_out2.reshape((2**j, 2, 2**(i - j - 1), 2, 2**(L - 1 - i), 2, 2, 2, 2))
        psi_out2 = psi_out2.transpose((0, 5, 2, 6, 4, 8, 7, 3, 1))
        psi_out2 = psi_out2.reshape(-1)

        file["psi_out2"] = interleave_complex(psi_out2, "cplx")

        file.close()


def main():
    apply_matchgate_data()
    apply_matchgate_backward_data()
    apply_matchgate_backward_array_data()
    apply_matchgate_to_array_data()
    apply_matchgate_placeholder_data()


if __name__ == "__main__":
    main()

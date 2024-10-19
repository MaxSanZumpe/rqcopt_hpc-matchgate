import numpy as np
import h5py
import os
import rqcopt_matfree as oc
from io_util import interleave_complex


def apply_matchgate_brickwall_unitary_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 8

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_apply_matchgate_brickwall_unitary_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        # random input statevector
        psi = oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, "cplx")

        max_nlayers = 4

        #general machgate blocks list and corresponding matrix list
        Vlist = []
        Mlist = [oc.crandn((2, 2), rng) for _ in range(2*max_nlayers)]
        for i in range(max_nlayers):
            file[f"v/center{i}"] = interleave_complex(Mlist[2*i    ], "cplx")
            file[f"v/corner{i}"] = interleave_complex(Mlist[2*i + 1], "cplx")
            Vlist.append(oc.matchgate_matrix(Mlist[2*i], Mlist[2*i + 1]))

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        for i, nlayers in enumerate([3, 4]):
            chi = oc.apply_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            file[f"chi{i}"] = interleave_complex(chi, "cplx")



def matchgate_brickwall_unitary_backward_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 6

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_brickwall_unitary_backward_cplx.hdf5')

    with h5py.File(file_path, "w") as file:
        # random input statevector
        psi = oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, "cplx")

        max_nlayers = 4

        #general machgate blocks list and corresponding matrix list
        Vlist = []
        Mlist = [oc.crandn((2, 2), rng) for _ in range(2*max_nlayers)]
        for i in range(max_nlayers):
            file[f"v/center{i}"] = interleave_complex(Mlist[2*i    ], "cplx")
            file[f"v/corner{i}"] = interleave_complex(Mlist[2*i + 1], "cplx")
            Vlist.append(oc.matchgate_matrix(Mlist[2*i], Mlist[2*i + 1]))

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        for i, nlayers in enumerate([3, 4]):
            psi_out = oc.apply_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            file[f"psi_out{i}"] = interleave_complex(psi_out, "cplx")

        # fictitious upstream derivatives
        dpsi_out = oc.crandn(2**L, rng)
        file["dpsi_out"] = interleave_complex(dpsi_out, "cplx")


def matchgate_brickwall_unitary_backward_hessian_data():

    # random number generator
    rng = np.random.default_rng(47)

    # system size
    L = 6

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_brickwall_unitary_backward_hessian_cplx.hdf5')

    with h5py.File(file_path, "w") as file:
        # random input statevector
        psi = oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, "cplx")

        max_nlayers = 4

        #general machgate blocks list and corresponding matrix list
        Vlist = []
        Mlist = [oc.crandn((2, 2), rng) for _ in range(2*max_nlayers)]
        for i in range(max_nlayers):
            file[f"v/center{i}"] = interleave_complex(Mlist[2*i    ], "cplx")
            file[f"v/corner{i}"] = interleave_complex(Mlist[2*i + 1], "cplx")
            Vlist.append(oc.matchgate_matrix(Mlist[2*i], Mlist[2*i + 1]))

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        for i, nlayers in enumerate([1, 2, 3, 4]):
            psi_out = oc.apply_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            file[f"psi_out{i}"] = interleave_complex(psi_out, "cplx")

        # fictitious upstream derivatives
        dpsi_out = oc.crandn(2**L, rng)
        file["dpsi_out"] = interleave_complex(dpsi_out, "cplx")

        # gradient direction of quantum gates
        ZMlist = [0.5 * oc.crandn((2, 2), rng) for _ in range(2*max_nlayers)]
        for i in range(max_nlayers):
            file[f"z/center{i}"] = interleave_complex(ZMlist[2*i    ], "cplx")
            file[f"z/corner{i}"] = interleave_complex(ZMlist[2*i + 1], "cplx")


def main():
    apply_matchgate_brickwall_unitary_data()
    matchgate_brickwall_unitary_backward_data()
    matchgate_brickwall_unitary_backward_hessian_data()


if __name__ == "__main__":
    main()

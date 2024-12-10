import os
import sys
import h5py
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm 

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io_util as io
import rqcopt_matfree as oc
import permutations as p
import sparse_targets as st


def extract_matchgate(V):
    V = np.roll(V, (-1, -1), (0, 1))  
    G1 = V[0:2, 0:2]
    G2 = np.roll(V[2:, 2:], (-1, -1), (0, 1))

    return G1, G2

def construct_hubbard1d_kinetic_term(J):
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)

    return -J*hop

def construct_hubbard1d_interac_term(g):
    n = np.array([[0, 0], [0, 1]])

    return g*np.kron(n, n)



L = 8
nqubits = 2*L
J = 1

g = 4.0
t = 1

method = "suzuki"

us = 13
order = 4

H = st.construct_sparse_hubbard1d_hamiltonian(L, J, g)

psi0 = np.ones(2**nqubits)
psi0 /= np.linalg.norm(psi0)

Upsi = sp.linalg.expm_multiply(-1j*H*t, psi0)

# Change the splitting method here
splitting = oc.SplittingMethod.suzuki(3, order/2)
h_kin = construct_hubbard1d_kinetic_term(J)
h_int = construct_hubbard1d_interac_term(g)

terms = [h_kin, h_kin, h_int]

uindex, coeffs_ulist = oc.merge_layers(us*splitting.indices, us*splitting.coeffs)
ulayers = len(coeffs_ulist)


file_dir  = os.path.dirname(__file__)
file_path = os.path.join(file_dir, "error_in" ,f"hubbard1d_{method}{order}_q{nqubits}_us{us}_u{ulayers}_t{t:.2f}s_g{g:.2f}_error_in.hdf5")

with h5py.File(file_path, "w") as file:
    
    file["psi0"] = io.interleave_complex(psi0, "cplx")
    file["Upsi"] = io.interleave_complex(Upsi, "cplx")


    for s in range(1, us + 1):

        uindex, coeffs_ulist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)

        layers = len(coeffs_ulist)

        dt = t/s
        ulist  = [expm(-1j*c*dt*terms[i]) for c, i in zip(coeffs_ulist, uindex)]

        uperms = p.permuations.hubbard1d(uindex, nqubits).perm_list

        assert(len(uperms) == layers)
        assert(len(list(uperms[1])) == nqubits)

        ublocks = []
        for X in ulist:
            X1, X2 = extract_matchgate(X)
            ublocks.append(X1)
            ublocks.append(X2)

        ublocks = np.array(ublocks)

        file[f"ulist_{s}"] = io.interleave_complex(np.stack(ublocks), "cplx")

        for i in range(layers):
            file[f"uperm{i}_{s}"] = np.arange(nqubits) if uperms[i] is None else uperms[i]
            file.attrs[f"layers_{s}"] = layers


import os
import h5py
import numpy as np
import io_util as io
import rqcopt_matfree as oc
from scipy.linalg import expm
import permutations

def crandn(size, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)

def extract_matchgate(V):
    V = np.roll(V, (-1, -1), (0, 1))  
    G1 = V[0:2, 0:2]
    G2 = np.roll(V[2:, 2:], (-1, -1), (0, 1))

    return G1, G2

def construct_spl_hubbard_local_term(J, U):
    n = np.array([[0, 0], [0, 1]])
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)
    int = np.kron(n, n)

    return -J*hop + U*int


nqubits = 10
J = 1

U = 4.0
t = 0.01

s = 5
us = 40

order = 2

splitting = oc.SplittingMethod.suzuki(2, order/2)
hloc = construct_spl_hubbard_local_term(J, U)
vindex, coeffs_vlist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)
uindex, coeffs_ulist = oc.merge_layers(us*splitting.indices, us*splitting.coeffs)

nlayers = len(coeffs_vlist)
ulayers = len(coeffs_ulist)

vlist  = [expm(-1j*c*t*hloc) for c in coeffs_vlist]
ulist  = [expm(-1j*c*t*hloc) for c in coeffs_ulist]

perms  = permutations.permuations.spl_hubbard1d(vindex, nqubits).perm_list
uperms = permutations.permuations.spl_hubbard1d(uindex, nqubits).perm_list

print(perms)

assert(len(perms)  == nlayers) 
assert(len(uperms) == ulayers)

ublocks = []
for X in ulist:
    X1, X2 = extract_matchgate(X)
    ublocks.append(X1)
    ublocks.append(X2)

ublocks = np.array(ublocks)

vblocks = []
for V in vlist:
    V1, V2 = extract_matchgate(V)
    vblocks.append(V1)
    vblocks.append(V2)

vblocks = np.array(vblocks)

file_dir  = os.path.dirname(__file__)
file_path = os.path.join(file_dir, "input" ,f"spl_hubbard1d_suzuki{order}_n{nlayers}_q{nqubits}_u{ulayers}_t{t}s_g{U}_init.hdf5")

# save initial data to disk
with h5py.File(file_path, "w") as file:

    
    rng = np.random.default_rng(182)
    psi = crandn(2**nqubits, rng)
    psi /= np.linalg.norm(psi)
    file["psi"] = io.interleave_complex(psi, "cplx")

    file[f"ulist"] = io.interleave_complex(np.stack(ublocks), "cplx")

    for i in range(ulayers):
        file[f"uperm{i}"] = np.arange(nqubits) if uperms[i] is None else uperms[i]

    file[f"vlist"] = io.interleave_complex(np.stack(vblocks), "cplx")

    for i in range(nlayers):
        file[f"perm{i}"] = np.arange(nqubits) if perms[i] is None else perms[i]

    # store parameters
    file.attrs["nqubits"] = nqubits
    file.attrs["nlayers"] = nlayers
    file.attrs["ulayers"] = ulayers
    file.attrs["J"] = float(J)
    file.attrs["U"] = float(U)
    file.attrs["t"] = float(t)

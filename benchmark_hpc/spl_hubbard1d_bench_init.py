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


nqubits = 8
J = 1

g = 1.5
t = 1

s = 1
us = 12

dt = t/s
udt = t/us

order = 2

splitting = oc.SplittingMethod.suzuki(2, order/2)
usplitting = oc.SplittingMethod.suzuki(2, 3)
hloc = construct_spl_hubbard_local_term(J, g)
vindex, coeffs_vlist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)
uindex, coeffs_ulist = oc.merge_layers(us*usplitting.indices, us*usplitting.coeffs)

nlayers = len(coeffs_vlist)
ulayers = len(coeffs_ulist)

vlist  = [expm(-1j*c*dt*hloc) for c in coeffs_vlist]
ulist  = [expm(-1j*c*udt*hloc) for c in coeffs_ulist]


perms  = permutations.permuations.spl_hubbard1d(vindex, nqubits).perm_list
uperms = permutations.permuations.spl_hubbard1d(uindex, nqubits).perm_list


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

script_dir = os.path.dirname(__file__)
file_dir = os.path.join(script_dir, f"bench_in/q{nqubits}")

if not os.path.exists(file_dir):
        os.makedirs(file_dir)

if not os.path.exists(os.path.join(script_dir, f"bench_out/q{nqubits}")):
        os.makedirs(os.path.join(script_dir, f"bench_out/q{nqubits}"))

file_path = os.path.join(file_dir,f"n{nlayers}_q{nqubits}_u{ulayers}_bench_in.hdf5")

# save initial data to disk
with h5py.File(file_path, "w") as file:

    rng = np.random.default_rng(182)
    psi = np.ones(2**nqubits)
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
    file.attrs["g"] = float(g)
    file.attrs["t"] = float(t)

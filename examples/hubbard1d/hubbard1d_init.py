import os
import h5py
import numpy as np
import io_util as io
import rqcopt_matfree as oc
from scipy.linalg import expm
import permutations
import sparse_targets as st

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

def construct_hubbard_kinetic_term(J):
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)

    return -J*hop

def construct_hubbard_interac_term(U):
    n = np.array([[0, 0], [0, 1]])

    return U*np.kron(n, n)

L = 4
nqubits = 2*L
J = 1

full_matrix = True

if (full_matrix):
    assert(nqubits <= 12)

g = 4.0
t = 0.2

s = 1
us = 1


if (full_matrix):
    assert(us == 1)

dt = t/s
udt = t/us

model = "suzuki4"

match model:
    case "suzuki2":
        order = 2
        splitting = oc.SplittingMethod.suzuki(3, order/2)
    case "yoshida4":
        order = 4
        splitting = oc.SplittingMethod.yoshida4(3)
    case "suzuki4":
        order = 4
        splitting = oc.SplittingMethod.suzuki(3, order/2)
    case "auzinger6":
        order = 6
        splitting = oc.SplittingMethod.auzinger15_6()
    case "suzuki6":
        order = 6
        splitting = oc.SplittingMethod.suzuki(3, order/2)


usplitting = oc.SplittingMethod.auzinger15_6()

h_kin = construct_hubbard_kinetic_term(J)
h_int = construct_hubbard_interac_term(g)

terms = [h_kin, h_kin, h_int]

vindex, coeffs_vlist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)
uindex, coeffs_ulist = oc.merge_layers(us*usplitting.indices, us*usplitting.coeffs)

nlayers = len(coeffs_vlist)
ulayers = len(coeffs_ulist)


vlist  = [expm(-1j*c*dt*terms[i]) for c, i in zip(coeffs_vlist, vindex)]
ulist  = [expm(-1j*c*udt*terms[i]) for c, i in zip(coeffs_ulist, uindex)]

perms  = permutations.permuations.hubbard1d(vindex, nqubits).perm_list
uperms = permutations.permuations.hubbard1d(uindex, nqubits).perm_list

assert(len(perms)  == nlayers) 
assert(len(uperms) == ulayers)

assert(len(list(perms[1])) == nqubits)
assert(len(list(uperms[1])) == nqubits)

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

script_dir  = os.path.dirname(__file__)
file_dir  = os.path.join(script_dir, f"opt_in/q{nqubits}")

if not os.path.exists(file_dir):
        os.makedirs(file_dir)

if not os.path.exists(os.path.join(script_dir, f"opt_out/q{nqubits}")):
        os.makedirs(os.path.join(script_dir, f"opt_out/q{nqubits}"))

if (full_matrix == True):
    file_path = os.path.join(file_dir, f"hubbard1d_q{nqubits}_unitary_t{t:.2f}s_g{g:.2f}_init.hdf5")

    expiH = st.hubbard1d_unitary(L, J, g, t)
    with h5py.File(file_path, "w") as file:
        file["expiH"] = io.interleave_complex(expiH, "cplx")

file_path = os.path.join(file_dir, f"hubbard1d_{model}_n{nlayers}_q{nqubits}_u{ulayers}_t{t:.2f}s_g{g:.2f}_init.hdf5")

# save initial data to disk
with h5py.File(file_path, "w") as file:

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

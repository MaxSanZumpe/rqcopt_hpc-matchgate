import os
import h5py
import numpy as np
import io_util as io
import rqcopt_matfree as oc
from scipy.linalg import expm
import permutations as ps

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

def main():

    Lx = 4
    Ly = 4
    nqubits = int(Lx*Ly)

    # parameters for hamiltonian
    J = 1
    g = 1.5
    t = 0.25

    s = 2  #number of splitting steps
    us = 4
    

    dt = t/s
    udt = t/us

    model = "suzuki2"
    
    match model:
        case "suzuki2":
            order = 2
            splitting = oc.SplittingMethod.suzuki(4, order/2)

        case "suzuki4":
            order = 4
            splitting = oc.SplittingMethod.suzuki(4, order/2)

    usplitting = oc.SplittingMethod.suzuki(4, 3)

    h = construct_spl_hubbard_local_term(J, g)
    terms = [h, h, h, h]

    vindex, coeffs_vlist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)
    uindex, coeffs_ulist = oc.merge_layers(us*usplitting.indices, us*usplitting.coeffs)

    
    nlayers = len(coeffs_vlist)
    ulayers = len(coeffs_ulist) 

    
    vlist  = [expm(-1j*c*dt*terms[i]) for c, i in zip(coeffs_vlist, vindex)]
    ulist  = [expm(-1j*c*udt*terms[i]) for c, i in zip(coeffs_ulist, uindex)]

    perms  = ps.permuations.spl_hubbard2d(vindex, Lx, Ly).perm_list
    uperms = ps.permuations.spl_hubbard2d(uindex, Lx, Ly).perm_list   


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
    file_path = os.path.join(file_dir, "opt_in" ,f"spl_hubbard2d_{model}_n{nlayers}_q{nqubits}_u{ulayers}_t{t:.2f}s_g{g:.2f}_init.hdf5")

    # save initial data to disk
    with h5py.File(file_path, "w") as file:

        file["nqubits"] = nqubits

        file["nlayers"] = nlayers

        file["ulayers"] = ulayers

        file[f"ulist"] = io.interleave_complex(np.stack(ublocks), "cplx")

        for i in range(ulayers):
            file[f"uperm{i}"] = np.arange(nqubits) if uperms[i] is None else uperms[i]

        file[f"vlist"] = io.interleave_complex(np.stack(vblocks), "cplx")

        for i in range(nlayers):
            file[f"perm{i}"] = np.arange(nqubits) if perms[i] is None else perms[i]
        # store parameters
        file.attrs["L"] = nqubits
        file.attrs["J"] = float(J)
        file.attrs["g"] = float(g)
        file.attrs["t"] = float(t)

if __name__ == "__main__":
    main()

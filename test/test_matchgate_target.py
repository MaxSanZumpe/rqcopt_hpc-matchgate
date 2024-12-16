import numpy as np
from scipy.stats import ortho_group, unitary_group
import h5py
from io_util import interleave_complex
import rqcopt_matfree as oc
import os
import sparse_targets
import permutations as ps


def _ufunc_cplx(x):
    n = len(x)
    return np.array([  (-1.1 + 0.8j) * x[((i + 3) * 113) % n]
                     + ( 0.4 - 0.7j) * x[((i + 9) * 173) % n]
                     + ( 0.5 + 0.1j) * x[i]
                     + (-0.3 + 0.2j) * x[((i + 4) * 199) % n] for i in range(n)])


def matchgate_circuit_unitary_target_data():

    # random number generator
    rng = np.random.default_rng(495)

    # system size
    nqubits = 7
    # number of gates
    ngates  = 5

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_circuit_unitary_target_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        # general random 2x2 matrices (do not need to be unitary for this test)
        gate_blocks = [0.5 * oc.crandn((2, 2), rng) for _ in range(2 * ngates)]
        gates = []
        for i in range(ngates):
            file[f"G/center{i}"] = interleave_complex(gate_blocks[2*i    ], "cplx")
            file[f"G/corner{i}"] = interleave_complex(gate_blocks[2*i + 1], "cplx")
            gates.append(oc.matchgate_matrix(gate_blocks[2*i    ], gate_blocks[2*i + 1]))

        # random wires which the gates act on
        wires = np.array([rng.choice(nqubits, 2, replace=False) for _ in range(ngates)])
        file["wires"] = wires

        ufunc = _ufunc_cplx

        # target function value
        f = oc.circuit_opt_matfree._f_circuit_unitary_target_matfree(gates, wires, nqubits, ufunc)
        file["f"] = interleave_complex(f, "cplx")


def matchgate_brickwall_unitary_target_data():

    # random number generator
    rng = np.random.default_rng(41)

    # system size
    L = 8

    max_nlayers = 5

    
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_brickwall_unitary_target_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        # general random 2x2 matrices (do not need to be unitary for this test)
        V_blocks = [0.5 * oc.crandn((2, 2), rng) for _ in range(2 * max_nlayers)]
        Vlist = []
        for i in range(max_nlayers):
            file[f"V/center{i}"] = interleave_complex(V_blocks[2*i    ], "cplx")
            file[f"V/corner{i}"] = interleave_complex(V_blocks[2*i + 1], "cplx")
            Vlist.append(oc.matchgate_matrix(V_blocks[2*i    ], V_blocks[2*i + 1]))

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        ufunc = _ufunc_cplx

        for i, nlayers in enumerate([4, 5]):
            # target function value
            f = oc.brickwall_opt_matfree._f_brickwall_unitary_target_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"f{i}"] = interleave_complex(f, "cplx")


def _brickwall_unitary_plain_hessian_matrix_matfree(Vlist, L, Ufunc, perms):
    """
    Construct the Hessian matrix of Re tr[U† W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist,
    using the provided matrix-free application of U to a state.
    """
    n = len(Vlist)
    H = np.zeros((n, 16, n, 16), dtype=Vlist[0].dtype)
    for j in range(n):
        for k in range(16):
            # unit vector
            Z = np.zeros(16)
            Z[k] = 1
            Z = Z.reshape((4, 4))
            dVZj = oc.brickwall_unitary_hess_matfree(Vlist, L, Z, j, Ufunc, perms, unitary_proj=False)
            for i in range(n):
                H[i, :, j, k] = dVZj[i].reshape(-1)
    return H.reshape((n * 16, n * 16))


def matchgate_brickwall_unitary_target_gradient_hessian_data():

    # random number generator
    rng = np.random.default_rng(45)

    # system size
    L = 6

    max_nlayers = 5

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_brickwall_unitary_target_gradient_hessian_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        # general random 2x2 matrices (do not need to be unitary for this test)
        V_blocks = [0.5 * oc.crandn((2, 2), rng) for _ in range(2 * max_nlayers)]
        Vlist = []
        for i in range(max_nlayers):
            file[f"V/center{i}"] = interleave_complex(V_blocks[2*i    ], "cplx")
            file[f"V/corner{i}"] = interleave_complex(V_blocks[2*i + 1], "cplx")
            Vlist.append(oc.matchgate_matrix(V_blocks[2*i    ], V_blocks[2*i + 1]))

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        # gradient direction of quantum gates
        Z_blocks = [0.5 * oc.crandn((2, 2), rng) for _ in range(2 * max_nlayers)]
        Zlist = []
        for i in range(max_nlayers):
            file[f"Z/center{i}"] = interleave_complex(Z_blocks[2*i    ], "cplx")
            file[f"Z/corner{i}"] = interleave_complex(Z_blocks[2*i + 1], "cplx")
            Vlist.append(oc.matchgate_matrix(Z_blocks[2*i    ], Z_blocks[2*i + 1]))

        ufunc = _ufunc_cplx

        for i, nlayers in enumerate([4, 5]):
            # target function value
            f = oc.brickwall_opt_matfree._f_brickwall_unitary_target_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"f{i}"] = interleave_complex(f, "cplx")
            # gate gradients
            dVlist = -oc.brickwall_unitary_grad_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            for k, dV in enumerate(dVlist):
                A, B = oc.extract_matchgate(dV.conj())                      # complex conjugation due to different convention
                file[f"dVlist{i}/center{k}"] = interleave_complex(A, "cplx")   
                file[f"dVlist{i}/corner{k}"] = interleave_complex(B, "cplx")
            # Hessian matrix
            hess = -_brickwall_unitary_plain_hessian_matrix_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"hess{i}"] = interleave_complex(hess.conj(), "cplx")       # complex conjugation due to different convention

        file.close()


def matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix_data():

    # random number generator
    rng = np.random.default_rng(47)

    # system size
    L = 6

    # number of layers
    nlayers = 5

    ctype = "cplx"

    

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', f"test_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix_{ctype}.hdf5",)

    with h5py.File(file_path, "w") as file:

        # random unitaries (unitary property required for Hessian matrix to be symmetric)
        Vlist_center = [unitary_group.rvs(2, random_state=rng) for _ in range(nlayers)]
        Vlist_corner = [unitary_group.rvs(2, random_state=rng) for _ in range(nlayers)]
        Vlist = []
        for i in range(nlayers):
            Vlist.append(oc.matchgate_matrix(Vlist_center[i], Vlist_corner[i]))
            file[f"V/center{i}"] = interleave_complex(Vlist_center[i], ctype)
            file[f"V/corner{i}"] = interleave_complex(Vlist_corner[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(nlayers)]
        for i in range(nlayers):
            file[f"perm{i}"] = perms[i]

        ufunc = _ufunc_cplx

        # target function value
        f = oc.brickwall_opt_matfree._f_brickwall_unitary_target_matfree(Vlist, L, ufunc, perms)
        file["f"] = interleave_complex(f, ctype)

        # gate gradients as real vector
        grad = -oc.brickwall_unitary_gradient_vector_matfree(Vlist, L, ufunc, perms)

        match_grad = np.zeros(8 * nlayers)
        map = [4, None, None, 5, None, 0, 1, None, None, 2, 3, None, 6, None, None, 7]
        for i in range(nlayers):
            for j, m, in enumerate(map):
                if m != None:
                    match_grad[i*8 + m] = grad[i*16 + j]
        
        file["grad"] = match_grad

        #Hessian matrix
        H = -oc.brickwall_unitary_hessian_matrix_matfree(Vlist, L, ufunc, perms)

        match_H = np.zeros((8 * nlayers * 8 * nlayers))
        for i in range(nlayers):
            for j in range(nlayers):
                for x, m in enumerate(map):
                    if m != None:
                        for y, n in enumerate(map):
                            if n != None:
                                match_H[((i*8 + m)*nlayers + j)*8 + n] = H[i*16 + x][j*16 + y]
    
        
        print(match_H.shape)
        file["H"] = match_H


def matchgate_ti_brickwall_unitary_target_gradient_hessian_data():

    # random number generator
    rng = np.random.default_rng(48)

    # system size
    L = 8

    s = 3
    nlayers = 7

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_ti_brickwall_unitary_target_gradient_hessian_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        J = 1
        g = 4
        t = 1

        # Should be translationally invariant unitary matrix
        #expiH = sparse_targets.construct_sparse_hubbard1d_hamiltonian(4, J, g)
        expiH = sparse_targets.spl_hubbard1d_unitary(8, J, g, t)
        file["expiH"] = interleave_complex(expiH, "cplx")

        # general random 2x2 matrices (do not need to be unitary for this test)
        V_blocks = [0.5 * oc.crandn((2, 2), rng) for _ in range(2 * nlayers)]
        Vlist = []
        for i in range(nlayers):
            file[f"V/center{i}"] = interleave_complex(V_blocks[2*i    ], "cplx")
            file[f"V/corner{i}"] = interleave_complex(V_blocks[2*i + 1], "cplx")
            Vlist.append(oc.matchgate_matrix(V_blocks[2*i    ], V_blocks[2*i + 1]))
        
        splitting = oc.SplittingMethod.suzuki(2, 1)
        vindex, coeffs_vlist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)
        perms  = ps.permuations.spl_hubbard1d(vindex, L).perm_list

        for i in range(nlayers):
            file[f"perm{i}"] = np.arange(L) if perms[i] is None else perms[i]


def matchgate_ti_brickwall_unitary_target_gradient_hessian_data2():

    rng = np.random.default_rng(49)

    # system size
    L = 4
    q = 2*L

    s = 1
    nlayers = 5

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_ti_brickwall_unitary_target_gradient_hessian2_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        J = 1
        g = 4
        t = 1

        # Should be translationally invariant unitary matrix
        expiH = sparse_targets.hubbard1d_unitary(L, J, g, t)
        #expiH = sparse_targets.spl_hubbard1d_unitary(8, J, g, t)
        file["expiH"] = interleave_complex(expiH, "cplx")

        # general random 2x2 matrices (do not need to be unitary for this test)
        V_blocks = [0.5 * oc.crandn((2, 2), rng) for _ in range(2 * nlayers)]

        file[f"vlist"] = interleave_complex(np.stack(V_blocks), "cplx")

        
        splitting = oc.SplittingMethod.suzuki(3, 1)
        vindex, coeffs_vlist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)
        perms  = ps.permuations.hubbard1d(vindex, q).perm_list

        for i in range(nlayers):
            file[f"perm{i}"] = np.arange(q) if perms[i] is None else perms[i]


def matchgate_ti_brickwall_unitary_target_gradient_hessian_data3():
    from scipy.linalg import expm

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

    # system size
    L = 4
    q = 2*L

    nlayers = 7
    ulayers = 201

    s = 3
    us = 100

    
    J = 1
    g = 4
    t = 1

    dt = t/s
    udt = t/us

    splitting = oc.SplittingMethod.suzuki(2, 1)
    
    vindex, coeffs_vlist = oc.merge_layers(s*splitting.indices, s*splitting.coeffs)
    uindex, coeffs_ulist = oc.merge_layers(us*splitting.indices, us*splitting.coeffs)
    
    perms  = ps.permuations.spl_hubbard1d(vindex, q).perm_list
    uperms = ps.permuations.spl_hubbard1d(uindex, q).perm_list

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_matchgate_ti_brickwall_unitary_target_gradient_hessian3_cplx.hdf5')

    with h5py.File(file_path, "w") as file:

        h = construct_spl_hubbard_local_term(J, g)
        terms = [h, h]

        vlist  = [expm(-1j*c* dt*terms[i]) for c, i in zip(coeffs_vlist, vindex)]
        ulist  = [expm(-1j*c*udt*terms[i]) for c, i in zip(coeffs_ulist, uindex)]

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

        ublocks = np.array(vblocks)
        
        file[f"vlist"] = interleave_complex(np.stack(vblocks), "cplx")
        file[f"ulist"] = interleave_complex(np.stack(ublocks), "cplx")   

        for i in range(nlayers):
            file[f"perm{i}"] = np.arange(q) if perms[i] is None else perms[i]

        
        for i in range(ulayers):
            file[f"uperm{i}"] = np.arange(q) if uperms[i] is None else uperms[i]



def main():
    # matchgate_circuit_unitary_target_data()
    # matchgate_brickwall_unitary_target_data()
    # matchgate_brickwall_unitary_target_gradient_hessian_data()
    # matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix_data()
    matchgate_ti_brickwall_unitary_target_gradient_hessian_data()
    matchgate_ti_brickwall_unitary_target_gradient_hessian_data2()
    matchgate_ti_brickwall_unitary_target_gradient_hessian_data3()



if __name__ == "__main__":
    main()

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm


def construct_sparse_hubbard1d_kinetic_term(L):
    a = sp.csr_matrix(np.array([[0, 1], [0, 0]])) 
    k = sp.kron(a.T, a) + sp.kron(a, a.T)

    periodic = sp.kron(sp.kron(a, sp.eye(2**(L - 2))), a.T) + sp.kron(sp.kron(a.T, sp.eye(2**(L - 2))), a)
    periodic = sp.kron(periodic, sp.eye(2**L)) + sp.kron(sp.eye(2**L), periodic)

    kinetic = sp.kron(k, sp.eye(2**(L - 2))) + sp.kron(sp.eye(2**(L - 2)), k)

    for i in range(1, L - 2):
        kinetic += sp.kron(sp.kron(sp.eye(2**i), k), sp.eye(2**(L - i - 2)))

    kinetic = sp.kron(kinetic, sp.eye(2**L)) + sp.kron(sp.eye(2**L), kinetic)

    return kinetic + periodic


def kronZ(q):
    Z = sp.csr_matrix(np.array([[1, 0], [0, -1]]))
    kronZ = Z
    
    for _ in range(q - 1):
        kronZ = sp.kron(kronZ, Z)
    
    assert(kronZ.shape == (2**q, 2**q))
    return kronZ

def construct_sparse_exact_hubbard1d_kinetic_term(L):
    a = sp.csr_matrix(np.array([[0, 1], [0, 0]])) 
    k = sp.kron(a.T, a) + sp.kron(a, a.T)


    periodic = sp.kron(sp.kron(a, kronZ(L-2)), a.T) + sp.kron(sp.kron(a.T, kronZ(L-2)), a)
    periodic = sp.kron(periodic, sp.eye(2**L)) + sp.kron(sp.eye(2**L), periodic)

    kinetic = sp.kron(k, sp.eye(2**(L - 2))) + sp.kron(sp.eye(2**(L - 2)), k)

    for i in range(1, L - 2):
        kinetic += sp.kron(sp.kron(sp.eye(2**i), k), sp.eye(2**(L - i - 2)))

    kinetic = sp.kron(kinetic, sp.eye(2**L)) + sp.kron(sp.eye(2**L), kinetic)

    return kinetic + periodic


def construct_sparse_hubbard1d_interac_term(L):
    n = sp.csr_matrix(np.array([[0, 0], [0, 1]]))

    interac  = sp.kron(sp.kron(n, sp.eye(2**(L - 1))), sp.kron(n, sp.eye(2**(L - 1))))
    interac += sp.kron(sp.kron(sp.eye(2**(L - 1)), n), sp.kron(sp.eye(2**(L - 1)), n))

    for i in range(1, L - 1):
        interac += sp.kron(sp.kron(sp.kron(sp.eye(2**i), n), sp.eye(2**(L - i - 1))), sp.kron(sp.kron(sp.eye(2**i), n), sp.eye(2**(L - i - 1))))

    return interac
    

def construct_sparse_hubbard1d_hamiltonian(L, J, g):

    h1 = construct_sparse_hubbard1d_kinetic_term(L)
    h2 = construct_sparse_hubbard1d_interac_term(L)

    return -J*h1 + g*h2


def construct_sparse_exact_hubbard1d_hamiltonian(L, J, g):

    h1 = construct_sparse_exact_hubbard1d_kinetic_term(L)
    h2 = construct_sparse_hubbard1d_interac_term(L)

    return -J*h1 + g*h2


def construct_sparse_2d_2qubit_operator(i1, j1, i2, j2, Lx, Ly, a: sp.csr_matrix, b: sp.csr_matrix):
    L = int(Lx*Ly)
    p1 = int(j1 + Ly*i1)
    p2 = int(j2 + Ly*i2)

    if (p1 > p2):
        aux = p2
        p2 = p1
        p1 = aux

        aux = b
        b = a
        a = aux

    return sp.kron(sp.kron(sp.kron(sp.eye(2**p1), a), sp.eye(2**(p2 - p1 - 1))), sp.kron(b, sp.eye(2**(L - p2 - 1))))

def construct_sparse_spl_hubbard2d_local_kinetic_term(i1, j1, i2, j2, Lx, Ly):
    a = sp.csr_matrix(np.array([[0, 1], [0, 0]]))

    return construct_sparse_2d_2qubit_operator(i1, j1, i2, j2, Lx, Ly, a, a.T) + construct_sparse_2d_2qubit_operator(i1, j1, i2, j2, Lx, Ly, a.T, a)


def construct_sparse_spl_hubbard2d_local_interac_term(i1, j1, i2, j2, Lx, Ly):
    n = sp.csr_matrix(np.array([[0, 0], [0, 1]]))

    return construct_sparse_2d_2qubit_operator(i1, j1, i2, j2, Lx, Ly, n, n)

def construct_sparse_spl_hubbard2d_kinetic_term(Lx, Ly):
    L = Lx*Ly
    kinetic = 0*sp.eye(2**L)
    
    for i in range(Ly):
        for j in range(Ly):
            kinetic += construct_sparse_spl_hubbard2d_local_kinetic_term(i , j, i         , (j + 1)%Ly, Lx, Ly)
            kinetic += construct_sparse_spl_hubbard2d_local_kinetic_term(i , j, (i + 1)%Lx, j         , Lx, Ly)

    return kinetic

def construct_sparse_spl_hubbard2d_interac_term(Lx, Ly):
    L = Lx*Ly
    inter = 0*sp.eye(2**L)

    for i in range(Ly):
        for j in range(Ly):
            inter += construct_sparse_spl_hubbard2d_local_interac_term(i , j, i         , (j + 1)%Ly, Lx, Ly)
            inter += construct_sparse_spl_hubbard2d_local_interac_term(i , j, (i + 1)%Lx, j         , Lx, Ly)

    return inter

def construct_sparse_spl_hubbard2d_hamiltonian(Lx, Ly, J, g):
    
    h1 = construct_sparse_spl_hubbard2d_kinetic_term(Lx, Ly)
    h2 = construct_sparse_spl_hubbard2d_interac_term(Lx, Ly)

    return -J*h1 + g*h2


def construct_sparse_spl_hubbard1d_kinetic_term(L):
    a = sp.csr_matrix(np.array([[0, 1], [0, 0]])) 
    k = sp.kron(a.T, a) + sp.kron(a, a.T)

    
    periodic = sp.kron(sp.kron(a, sp.eye(2**(L - 2))), a.T) + sp.kron(sp.kron(a.T, sp.eye(2**(L - 2))), a)
    kinetic = sp.kron(k, sp.eye(2**(L - 2))) + sp.kron(sp.eye(2**(L - 2)), k)

    for i in range(1, L - 2):
        kinetic += sp.kron(sp.kron(sp.eye(2**i), k), sp.eye(2**(L - i - 2)))

    return kinetic + periodic


def construct_sparse_spl_hubbard1d_interac_term(L):
    n = sp.csr_matrix(np.array([[0, 0], [0, 1]]))
    v = sp.kron(n, n)

    interac = sp.kron(v, sp.eye(2**(L - 2))) + sp.kron(sp.eye(2**(L - 2)), v) + sp.kron(sp.kron(n, sp.eye(2**(L - 2))), n)

    for i in range(1, L - 2):
        interac += sp.kron(sp.kron(sp.eye(2**(i)), v), sp.eye(2**(L - i - 2)))

    return interac

def construct_sparse_spl_hubbard1d_hamiltonian(L, J, g):

    h1 = construct_sparse_spl_hubbard1d_kinetic_term(L)
    h2 = construct_sparse_spl_hubbard1d_interac_term(L)

    return -J*h1 + g*h2



def hubbard1d_unitary(L, J, g, t):
    h = construct_sparse_hubbard1d_hamiltonian(L, J, g).toarray()

    return expm(-1j*t*h)

def spl_hubbard1d_unitary(L, J, g, t):
    h = construct_sparse_spl_hubbard1d_hamiltonian(L, J, g).toarray()

    return expm(-1j*t*h)
    

def qubit_permutation(U: np.ndarray, qubit_dim: int, perm: list):
        L = qubit_dim
        gate = U.reshape((2,) * 2*L)
        perm_gate = np.transpose(gate, perm + [a + L for a in perm])
        perm_gate = perm_gate.reshape(2**L, 2**L)

        return perm_gate


def test_hubbard1d_and_spl_hubbard1d():
    def hubbard_hopping_term(L):
        a = np.array([[0, 1], [0, 0]])
        hop = np.kron(a.T, a) + np.kron(a, a.T)

        h = np.kron(hop, np.identity(2**(L - 2))) + np.kron(np.identity(2**(L - 2)), hop)

        periodic1 = np.kron(np.kron(a, np.identity(2**(L - 2))), a.T)
        periodic2 = np.kron(np.kron(a.T, np.identity(2**(L - 2))), a)
        periodic = periodic1 + periodic2 
        h += periodic

        for k in range(L - 3):
            h += np.kron(np.kron(np.identity(2**(k+1)), hop), np.identity(2**(L - k - 3)))
        
        spin_up = np.kron(h, np.identity(2**L))
        spin_down = np.kron(np.identity(2**L), h)

        return spin_up + spin_down
    
    def hubbard_interaction_term(L):
        n = np.array([[0, 0], [0, 1]])

        half_term_start = np.kron(n, np.identity(2**(L-1)))
        half_term_end = np.kron(np.identity(2**(L-1)), n)
        h = np.kron(half_term_start, half_term_start) + np.kron(half_term_end, half_term_end)

        for k in range(int(L - 2)):
            half_term = np.kron(np.kron(np.identity(2**(k + 1)), n), np.identity(2**(L - k - 2))) 
            h += np.kron(half_term, half_term)

        assert h.shape == (4**L, 4**L)
        return h
    
    def spinless_hubbard_hamiltonian(L, J, g):
        a = np.array([[0, 1], [0, 0]])
        n = np.array([[0, 0], [0, 1]])
        hop = np.kron(a, a.T) + np.kron(a.T, a)

        h_kinetic = np.zeros((2**L, 2**L), dtype=float)

        for k in range(L- 1):
            h_kinetic += np.kron(np.kron(np.identity(2**k), hop), np.identity(2**(L - k - 2)))

        h_repulsion = np.zeros((2**L, 2**L), dtype=float)

        for k in range(L - 1):
            h_repulsion += np.kron(np.kron(np.identity(2**k), np.kron(n, n)), np.identity(2**(L - k - 2)))

        periodic1 = np.kron(np.kron(a, np.identity(2**(L - 2))), a.T)
        periodic2 = np.kron(np.kron(a.T, np.identity(2**(L - 2))), a)
        periodic_kinetic   = periodic1 + periodic2
        periodic_repulsion = np.kron(np.kron(n, np.identity(2**(L - 2))), n) 

        h_kinetic   += periodic_kinetic
        h_repulsion += periodic_repulsion

        h = -J*h_kinetic + g*h_repulsion
        return h

    J = 1
    g = 4
    
    L = 4
    q = 2*L

    hopping = hubbard_hopping_term(L)
    interaction = hubbard_interaction_term(L)
    h_ref1 = -J*hopping + g*interaction
    h_ref2 = spinless_hubbard_hamiltonian(q, J, g)

    H1 = construct_sparse_hubbard1d_hamiltonian(L, J, g)
    H2 = construct_sparse_spl_hubbard1d_hamiltonian(q, J, g)

    H1 = H1.toarray()
    H2 = H2.toarray()


    if (not np.allclose(H1, h_ref1, 1e-10)):
        print("Hubbard1d hamiltonian failed.")
        
        
    if (not np.allclose(H2, h_ref2, 1e-10)):
        print(H2 - h_ref2)
        print("Spinless hubbard1d hamiltonian failed.")

    t = 1
    U1 = spl_hubbard1d_unitary(q, J, g, t)

    for shift in range(q):
        perm = list(np.roll(np.arange(q), shift))
        u_perm = qubit_permutation(U1, q, perm)

        if (not np.allclose(U1, u_perm)):
            print("Spinless hubbard1d unitary failed invariance")

    
    
    U2 = hubbard1d_unitary(L, J, g, t)

    for shift in range(L):
        perm = list(np.roll(np.arange(L), shift)) + list(np.roll(np.arange(L), shift) + 4)
        u_perm = qubit_permutation(U2, q, perm)
        if (not np.allclose(U2, u_perm)):
            print("Spinless hubbard1d unitary failed invariance")
def test_spl_hubbard2d():
    a = np.array([[0, 1], [0, 0]])
    n = sp.csr_matrix(np.array([[0, 0], [0, 1]]))
    k = sp.kron(a.T, a) + sp.kron(a, a.T)

    k_ref = sp.kron(sp.kron(sp.eye(2**(13)), k), sp.eye(2**1)) 
    k_test = construct_sparse_spl_hubbard2d_local_kinetic_term(3,1, 3, 2, 4, 4)

    print(k_ref - k_test)

    i1, j1 = (0, 0)
    i2, j2 = (0, 1)
    
    p1 = int(j1 + 4*i1)
    p2 = int(j2 + 4*i2)


    i_ref = sp.kron(sp.kron(sp.kron(sp.eye(2**0), n), sp.kron(sp.eye(2**(0)), n)), sp.eye(2**(14)))
    i_test = construct_sparse_spl_hubbard2d_local_interac_term(i1,j1, i2,j2, 4, 4)

    print(i_ref - i_test)

    h = construct_sparse_spl_hubbard2d_hamiltonian(4, 4, 1, 4)

if __name__ == "__main__":
    test_hubbard1d_and_spl_hubbard1d()
    test_spl_hubbard2d()


    


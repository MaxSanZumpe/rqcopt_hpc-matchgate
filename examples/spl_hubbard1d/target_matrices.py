import numpy as np
from scipy.linalg import expm 
    
def hubbard_hopping_term(L, boundary):

    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)

    h = np.kron(hop, np.identity(2**(L - 2))) + np.kron(np.identity(2**(L - 2)), hop)

    match boundary:
        case "periodic":
            periodic1 = np.kron(np.kron(a, np.identity(2**(L - 2))), a.T)
            periodic2 = np.kron(np.kron(a.T, np.identity(2**(L - 2))), a)
            periodic = periodic1 + periodic2 
            h += periodic
        case "open":
            pass
        case _:
            print("Boundary condition must be either open or periodic.")

    for k in range(L - 3):
        h += np.kron(np.kron(np.identity(2**(k+1)), hop), np.identity(2**(L - k - 3)))

    return h             

def spin_hubbard_hopping_term(qdim: int, boundary):
    L = int(qdim/2)
    half_term = hubbard_hopping_term(L, boundary)
    
    spin_up = np.kron(half_term, np.identity(2**L))
    spin_down = np.kron(np.identity(2**L), half_term)

    assert spin_up.shape == (4**L, 4**L)
    assert spin_down.shape == (4**L, 4**L)

    return spin_up + spin_down    
    
def spin_hubbard_interaction_term(qdim: int):
    L = int(qdim/2)
    n = np.array([[0, 0], [0, 1]])

    half_term_start = np.kron(n, np.identity(2**(L-1)))
    half_term_end = np.kron(np.identity(2**(L-1)), n)
    h = np.kron(half_term_start, half_term_start) + np.kron(half_term_end, half_term_end)

    for k in range(int(L - 2)):
        half_term = np.kron(np.kron(np.identity(2**(k + 1)), n), np.identity(2**(L - k - 2))) 
        h += np.kron(half_term, half_term)

    assert h.shape == (4**L, 4**L)
    return h

def spin_hubbard_unitary(qdim, J, U, t, boundary = "periodic"):
    
    assert qdim % 2 == 0

    hopping = spin_hubbard_hopping_term(qdim, boundary)
    interaction = spin_hubbard_interaction_term(qdim)
    h = -J*hopping + U*interaction

    return  expm(-1j * t * h)


def spinless_hubbard_unitary(qdim, J, U, t, boundary = "periodic"):
    a = np.array([[0, 1], [0, 0]])
    n = np.array([[0, 0], [0, 1]])
    hop = np.kron(a, a.T) + np.kron(a.T, a)

    h_kinetic = np.zeros((2**qdim, 2**qdim), dtype=float)

    for k in range(qdim - 1):
        h_kinetic += np.kron(np.kron(np.identity(2**k), hop), np.identity(2**(qdim - k - 2)))

    h_repulsion = np.zeros((2**qdim, 2**qdim), dtype=float)

    for k in range(qdim - 1):
        h_repulsion += np.kron(np.kron(np.identity(2**k), np.kron(n, n)), np.identity(2**(qdim - k - 2)))

    match boundary:
        case "periodic":
            periodic1 = np.kron(np.kron(a, np.identity(2**(qdim - 2))), a.T)
            periodic2 = np.kron(np.kron(a.T, np.identity(2**(qdim - 2))), a)
            periodic_kinetic   = periodic1 + periodic2
            periodic_repulsion = np.kron(np.kron(n, np.identity(2**(qdim - 2))), n) 

            h_kinetic   += periodic_kinetic
            h_repulsion += periodic_repulsion
        case "open":
            pass
        case _:
            print("Boundary condition must be either open or periodic.")

    h = -J*h_kinetic + U*h_repulsion
    return expm(-1j * t * h)

def invariant_unitary(qdim, t):
    single = np.array(np.array([[0, 1], [1, 0]]))

    h = single 

    for k in range(qdim - 1):
        h = np.kron(h, single)

    return h, expm(-1j * t * h)



import numpy as np
from scipy.linalg import expm
from permutations import spin_hubbart_double_strang_permutations_ccode, spin_hubbart_strang_trotter_permutations, spinless_hubbard_strang_trotter_permutations, brickwall_permutations

def extract_matchgate(V):
    V = np.roll(V, (-1, -1), (0, 1))  
    G1 = V[0:2, 0:2]
    G2 = np.roll(V[2:, 2:], (-1, -1), (0, 1))

    return G1, G2

def spin_hubbard_double_strang_init_ccode(J, U, t, s, qdim, mode: str):

    dt = t/s
    n = np.array([[0, 0], [0, 1]])
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)
    inter = np.kron(n, n)

    V1 = expm(1j * J * hop * dt/4)
    V2 = expm(1j * J * hop * dt/2)
    V3 = expm(-1j * U * inter * dt)

    G11, G12 = extract_matchgate(V1)
    G21, G22 = extract_matchgate(V2)
    G31, G32 = extract_matchgate(V3)

    perms, layers = spin_hubbart_double_strang_permutations_ccode(s, qdim)
   
    match mode:
        case "general":
            point_array = np.array([V1, V2, V1] + [V3, V1, V2, V2, V2, V1] * (s - 1) + [V3, V1, V2, V1])
        case "matchgate":
            point_array = np.array([G11, G12, G21, G22, G11, G12] + [G31, G32, G11, G12, G21, G22, G21, G22, G21, G22, G11, G12] * (s - 1) + [G31, G32, G11, G12, G21, G22, G11, G12])
        case _:
            print("Gate type undefined.")

    return point_array, perms, layers


def spin_hubbart_strang_trotter_init(J, U, t, s, qdim, mode: str):

    dt = t/s
    n = np.array([[0, 0], [0, 1]])
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)
    inter = np.kron(n, n)

    V1 = expm(1j * J * hop * dt/2)
    V2 = expm(1j * J * hop * dt)
    V3 = expm(-1j * U * inter * dt)

    G11, G12 = extract_matchgate(V1)
    G21, G22 = extract_matchgate(V2)
    G31, G32 = extract_matchgate(V3)

    perms, layers = spin_hubbart_strang_trotter_permutations(s, qdim)

    match mode:
        case "general":
            point_array = np.array([V1, V1] + [V3, V1, V2, V1] * (s - 1) + [V3, V1, V1])
        case "matchgate":
            point_array = np.array([G11, G12, G11, G12] + [G31, G32, G11, G12, G21, G22, G11, G12]*(s - 1) + [G31, G32, G11, G12, G11, G12])
        case _:
            print("Gate type undefined.")

    return point_array, perms, layers

def spinless_hubbard_trotter_init(J, U, t, s, qdim, mode: str):
    dt = t/s
    n = np.array([[0, 0], [0, 1]])
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)
    inter = np.kron(n, n)

    V = expm( 1j * (J * hop - U * inter)  * dt)

    G1, G2 = extract_matchgate(V)
    
    layers = 2*s
    perms = brickwall_permutations(layers, qdim)

    match mode:
        case "general":
            point_array = np.array([V]*2*s)
        case "matchgate":
            point_array = np.array([G1, G2]*2*s)
        case _:
            print("Gate type undefined.")

    return point_array, perms, layers



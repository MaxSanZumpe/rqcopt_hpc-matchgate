import numpy as np
from scipy.linalg import expm

def extract_matchgate(V):
    V = np.roll(V, (-1, -1), (0, 1))  
    G1 = V[0:2, 0:2]
    G2 = np.roll(V[2:, 2:], (-1, -1), (0, 1))

    return G1, G2

def spinless_hubbard_2d_order1_permutations(splitting_steps: int, Lx, Ly):
    L = Lx*Ly
    s = splitting_steps
    layers = 4*s

    assert (Lx == 4 and Ly == 4)

    even_to_odd_perm_horz = []
    even_to_odd_perm_vert = []
    even_to_even_perm_vert = []


    for j in range(Ly):
        for i in range(1, Lx, 2):
            even_to_odd_perm_horz  += [i + 4*j, 4*j + (i + 1) % Lx]


    for j in range(0, Ly, 2):
        for i in range(Lx):
            even_to_odd_perm_vert  += [(i + 4*(j + 1)), (i + 4*(j + 1) + 4) % L]
            even_to_even_perm_vert += [(i + 4*j), (i + 4*j + 4)]

    perms = [None, even_to_odd_perm_horz, even_to_even_perm_vert, even_to_odd_perm_vert]*s

    assert len(perms) == layers, f"got {len(perms)}"

    return perms, layers

def spinless_hubbard_2d_order2_permutations(splitting_steps: int, Lx, Ly):
    L = Lx*Ly
    s = splitting_steps
    layers = 6*s + 1

    assert (Lx == 4 and Ly == 4)
    
    even_to_even_perm_vert = []

    even_to_odd_perm_horz = []
    even_to_odd_perm_vert = []


    for j in range(Ly):
        for i in range(1, Lx, 2):
            even_to_odd_perm_horz  += [i + 4*j, 4*j + (i + 1) % Lx]


    for j in range(0, Ly, 2):
        for i in range(Lx):
            even_to_odd_perm_vert  += [(i + 4*(j + 1)), (i + 4*(j + 1) + 4) % L]
            even_to_even_perm_vert += [(i + 4*j), (i + 4*j + 4)]

    perms = [None, even_to_odd_perm_horz, even_to_even_perm_vert, even_to_odd_perm_vert, even_to_even_perm_vert, even_to_odd_perm_horz]*s + [None]

    assert len(perms) == layers, f"got {len(perms)}"

    return perms, layers

def spinless_hubbard_2d_trotter_order1_init(J, U, t, s, Lx, Ly):
    dt = t/s
    n = np.array([[0, 0], [0, 1]])
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)
    inter = np.kron(n, n)

    V = expm( -1j * (-J * hop + U * inter)  * dt )
    
    perms, layers = spinless_hubbard_2d_order1_permutations(s, Lx, Ly)

    G1, G2 = extract_matchgate(V)
    gate_array = np.array([G1, G2]*4*s)

    return gate_array, perms, layers


def spinless_hubbard_2d_trotter_order2_init(J, U, t, s, Lx, Ly):
    dt = t/s
    n = np.array([[0, 0], [0, 1]])
    a = np.array([[0, 1], [0, 0]])
    hop = np.kron(a.T, a) + np.kron(a, a.T)
    inter = np.kron(n, n)

    V1 = expm( -1j * (-J * hop + U * inter)  * dt/2 )
    V2 = expm( -1j * (-J * hop + U * inter)  * dt )
    
    perms, layers = spinless_hubbard_2d_order2_permutations(s, Lx, Ly)

    G11, G12 = extract_matchgate(V1)
    G21, G22 = extract_matchgate(V2)

    gate_array = [G11, G12] + [G21, G22]*5 + [G21, G22]*6*(s - 1) + [G11, G12] 
    gate_array = np.array(gate_array)

    return gate_array, perms, layers



        
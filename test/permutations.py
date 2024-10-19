import numpy as np 

def brickwall_permutations(layers: int, qubit_dim: int):

    perms = []
    even_to_odd_perm = [qubit_dim - 1] + [a for a in range(qubit_dim - 1)]

    for i in range(layers):
        if i % 2 == 0:
            perms.append(None)
        else:
            perms.append(even_to_odd_perm)
    
    return perms

def generate_permutation_list_hubbart(layers: int, qubit_dim: int):

    perms = []
    even_to_odd_perm = [int(qubit_dim/2) - 1] + [a for a in range(int(qubit_dim/2) - 1)] + \
                       [qubit_dim - 1] + [a for a in range(int(qubit_dim/2), qubit_dim - 1)]

    for i in range(layers):
        if i % 2 == 0:
            perms.append(None)
        else:
            perms.append(even_to_odd_perm)
    
    return perms

def spin_hubbart_double_strang_permutations_ccode(s: int, qubit_dim: int):
    L = qubit_dim
    half = int(L/2)
    layers = 6*s + 1

    interaction_perm = [int(a/2) if a % 2 == 0 else (int(a/2) + half) for a in range(L)]
    # even_to_odd_perm = [int(L/2) - 1] + [a for a in range(int(L/2) - 1)] +\
    #                    [L - 1] + [a for a in range(int(L/2), L - 1)]
    
    even_to_odd_perm = [a - 1 for a in range(2, int(L/2))    ] + [half - 1, 0] +\
                       [a - 1 for a in range(int(L/2) + 2, L)] + [L - 1, half]

    perms_endpoints = [None, even_to_odd_perm, None]
    perms_step = [interaction_perm, None, even_to_odd_perm, None, even_to_odd_perm, None]

    perms = perms_endpoints+ perms_step*(s-1) + [interaction_perm] + perms_endpoints

    assert len(perms) == layers, f"got {len(perms)}"

    return perms, layers

def spin_hubbard_interaction_term(lattice_length: int):
    L = lattice_length
    n = np.array([[0, 0], [0, 1]])

    half_term_start = np.kron(n, np.identity(2**(L-1)))
    half_term_end = np.kron(np.identity(2**(L-1)), n)
    h = np.kron(half_term_start, half_term_start) + np.kron(half_term_end, half_term_end)

    for k in range(int(L - 2)):
        half_term = np.kron(np.kron(np.identity(2**(k + 1)), n), np.identity(2**(L - k - 2))) 
        h += np.kron(half_term, half_term)

    assert h.shape == (4**L, 4**L)
    return h

def spin_hubbart_strang_trotter_permutations(splitting_steps: int, qubit_dim):
    L = qubit_dim
    s = splitting_steps
    layers = 4*s + 1

    interaction_perm =[a for a in range(L) if a % 2 == 0] + [a for a in range(L) if a % 2 != 0]
    even_to_odd_perm = [int(L/2) - 1] + [a for a in range(int(L/2) - 1)] + \
                       [L - 1] + [a for a in range(int(L/2), L - 1)]

    perms_endpoints = [None, even_to_odd_perm]
    perms_step = [interaction_perm, None, even_to_odd_perm, None]

    perms = perms_endpoints+ perms_step*(s-1) + [interaction_perm] + perms_endpoints

    assert len(perms) == layers, f"got {len(perms)}"

    return perms, layers


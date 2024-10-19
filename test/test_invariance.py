import numpy as np
from target_matrices import spinless_hubbard_unitary, invariant_unitary

def qubit_permutation(U: np.ndarray, qubit_dim: int, perm: list):

    L = qubit_dim
    gate = U.reshape((2,) * 2*L)
    perm_gate = np.transpose(gate, perm + [a + L for a in perm])
    perm_gate = perm_gate.reshape(2**L, 2**L)

    return perm_gate

q = 4
u = spinless_hubbard_unitary(q, 1, 2, 1, boundary = "periodic")
print(u.shape)

shift = 1
perm = list(np.roll(np.arange(q), shift))
print(perm + [a + q for a in perm])
u_perm = qubit_permutation(u, q, perm)
print(np.allclose(u, u_perm))
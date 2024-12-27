import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sparse_targets as st


def crandn(size, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


script_dir = os.path.dirname(__file__)

L_list = [4, 6, 8, 10]
J = 1
g = 4
t = 1

rng = np.random.default_rng(182)

errors = []
for L in L_list:

    h     = st.construct_sparse_hubbard1d_hamiltonian(L, J, g)
    h_ref = st.construct_sparse_exact_hubbard1d_hamiltonian(L, J, g)

    q = 2*L

    temp = []

    for _ in range(5):
        psi0 = crandn(2**q)
        psi0 /= np.linalg.norm(psi0)

        psi = sp.linalg.expm_multiply(-1j*h*t, psi0)
        psi_ref = sp.linalg.expm_multiply(-1j*h_ref*t, psi0)
        
        error = np.linalg.norm(psi_ref - psi, ord=np.inf)
        temp.append(error)

    errors.append(np.mean(np.array(temp)))

errors = np.array(errors)
x = 2*np.array(L_list)

fig, ax = plt.subplots()
ax.plot(x, errors, "^")


ax.set_title("Jordan-Wigner boundary term error")
ax.set_xlabel("Qubits")
ax.set_ylabel("$|| \psi_{approx.} - \psi_{exact} ||_{\infty}$")
fig.savefig(os.path.join(script_dir, f"plots/hubb1d_boundary.png"))



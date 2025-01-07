import os
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sparse_targets as st


def sparse_local_single_op(U, j, L):
    assert U.shape == (2, 2)
    return sp.kron(sp.eye(2**j), sp.kron(U, sp.eye(2**(L-j-1))))

L = 7
J = 1
g = 4.0

q = 2*L

H = st.construct_sparse_hubbard1d_hamiltonian(L, J, g)

psi0 = np.ones(2**q)
psi0 /= np.linalg.norm(psi0)

num = sp.csr_matrix([[0, 0], [0, 1]], dtype=float)


# total spin-up number operator
N_up = sum(sparse_local_single_op(num, j, 2*L) for j in range(0, L))
# total spin-down number operator
N_dn = sum(sparse_local_single_op(num, j, 2*L) for j in range(L, 2*L))

# 〈n_j↑〉
navr_up = np.vdot(psi0, sparse_local_single_op(num, 0, 2*L) @ psi0)
# 〈n_j↓〉
navr_dn = np.vdot(psi0, sparse_local_single_op(num, L, 2*L) @ psi0)

# n_j↑ measurement operators
meas_up_ops = [sparse_local_single_op(num - navr_up*np.identity(2), j, 2*L) for j in range(0, L)]
# n_j↓ measurement operators
meas_dn_ops = [sparse_local_single_op(num - navr_dn*np.identity(2), j, 2*L) for j in range(L, 2*L)]

Δt = 0.02
nsteps = 200

# store measurement average values
corr = np.zeros((L, nsteps+1), dtype=complex)

psit = psi0.copy()
# apply n_j↑ to ψ0
npsi = sparse_local_single_op(num - navr_up*np.identity(2), 0, 2*L) @ psi0
print(f"np.vdot(ψ0, nψ): {np.vdot(psi0, npsi)} (should be zero)")

for s in range(nsteps + 1):
    for j in range(L):
        corr[j, s] = np.vdot(psit, meas_up_ops[j] @ npsi)
    # wavefunctions at next time step
    psit = sp.linalg.expm_multiply(-1j*H*Δt, psit)
    npsi = sp.linalg.expm_multiply(-1j*H*Δt, npsi)

print(corr.shape)

# visualize dynamical correlation functions
script_dir = os.path.dirname(__file__)
vel = 1.5

plt.imshow(np.roll(corr, shift=(L-1)//2, axis=0).real.T,
            interpolation="nearest", aspect="auto",
            origin="lower", extent=(-L//2+0.5, L//2+0.5, 0, Δt*nsteps))
plt.xlabel("$\Delta j$", fontsize=12)
plt.ylabel("t (s)", fontsize=12)

plt.title(fr"$\langle \psi | n_{{j\uparrow}}(t) n_{{1\uparrow}}(0) | \psi \rangle$ for $H_{{hubb}}; U={g}; $vel: {vel} $s^{{-1}}$")
cbar = plt.colorbar()
plt.plot([ 0.5,  1+L//2], [0, L//2*1/vel], "w")
plt.plot([-0.5, -L//2], [0, L//2*1/vel], "w")
plt.savefig(os.path.join(script_dir, f"plots/hubb_g{g:.2f}_v{vel}_lightcone_py.png"), dpi = 300)

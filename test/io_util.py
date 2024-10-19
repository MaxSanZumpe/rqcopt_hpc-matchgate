import numpy as np


def interleave_complex(a: np.ndarray, ctype: str):
    """
    Interleave real and imaginary parts of a complex-valued array (for saving it to disk).
    """
    return a if ctype == "real" else np.stack((a.real, a.imag), axis=-1)


def crandn(size, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def extract_matchgate(V):
    V = np.roll(V, (-1, -1), (0, 1))  
    G1 = V[0:2, 0:2]
    G2 = np.roll(V[2:, 2:], (-1, -1), (0, 1))

    return G1, G2




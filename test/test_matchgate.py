import numpy as np
import h5py
from scipy.stats import unitary_group
import rqcopt_matfree as oc
from io_util import interleave_complex, crandn, extract_matchgate
import os


def real_to_tangent_matchgate_data():

    rng = np.random.default_rng(41)

    r = 0.5 * rng.standard_normal((8,))
    v1 = unitary_group.rvs(2, random_state=rng)
    v2 = unitary_group.rvs(2, random_state=rng)
    t1 = crandn((2, 2), rng)
    t2 = crandn((2, 2), rng)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_real_to_tangent_matchgate_cplx.hdf5')

    with h5py.File(file_path, "w") as file:
        file["r"] = r
        file["v/center"] = interleave_complex(v1, "cplx")
        file["v/corner"] = interleave_complex(v2, "cplx")
        file["t/center"] = interleave_complex(t1, "cplx")
        file["t/corner"] = interleave_complex(t2, "cplx")

def multiply_data_matchgate():

    rng = np.random.default_rng(53)

    a1 = crandn((2, 2), rng)
    a2 = crandn((2, 2), rng)

    b1 = crandn((2, 2), rng)
    b2 = crandn((2, 2), rng)

    c1 = a1 @ b1
    c2 = a2 @ b2

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_multiply_matchgate_cplx.hdf5')

    with h5py.File(file_path, "w") as file:
        file["a/center"] = interleave_complex(a1, "cplx")
        file["a/corner"] = interleave_complex(a2, "cplx")
        file["b/center"] = interleave_complex(b1, "cplx")
        file["b/corner"] = interleave_complex(b2, "cplx")
        file["c/center"] = interleave_complex(c1, "cplx")
        file["c/corner"] = interleave_complex(c2, "cplx")


def project_tangent_matchgate_data():

    # random number generator
    rng = np.random.default_rng(43)

    u1 = crandn((2, 2), rng)
    u2 = crandn((2, 2), rng)

    z1 = crandn((2, 2), rng)
    z2 = crandn((2, 2), rng)

    u = oc.matchgate_matrix(u1, u2)
    z = oc.matchgate_matrix(z1, z2)

    p = oc.project_tangent(u, z)
    p1, p2 = oc.extract_matchgate(p)

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'test_project_tangent_matchgate_cplx.hdf5')

    with h5py.File(file_path, "w") as file:
            file["u/center"] = interleave_complex(u1, "cplx")
            file["u/corner"] = interleave_complex(u2, "cplx")
            file["z/center"] = interleave_complex(z1, "cplx")
            file["z/corner"] = interleave_complex(z2, "cplx")
            file["p/center"] = interleave_complex(p1, "cplx")
            file["p/corner"] = interleave_complex(p2, "cplx")


def polar_matchgate_factor_data():

    # random number generator
    rng = np.random.default_rng(44)

    a1 = crandn((2, 2), rng)
    a2 = crandn((2, 2), rng)
    
    a = oc.matchgate_matrix(a1, a2)
    u, _ = oc.polar_decomp(a)

    u1, u2 = oc.extract_matchgate(u)

    # save to disk
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', "test_polar_matchgate_factor_cplx.hdf5")

    with h5py.File(file_path, "w") as file:
        file["a/center"] = interleave_complex(a1, "cplx")
        file["a/corner"] = interleave_complex(a2, "cplx")
        file["u/center"] = interleave_complex(u1, "cplx")
        file["u/corner"] = interleave_complex(u2, "cplx")


if __name__ == "__main__":
    multiply_data_matchgate()
    real_to_tangent_matchgate_data()
    project_tangent_matchgate_data()
    polar_matchgate_factor_data()
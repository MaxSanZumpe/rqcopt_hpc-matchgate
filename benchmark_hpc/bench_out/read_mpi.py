import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

nqubits = 12
nlayers = 5
ulayers = 253

script_dir = os.path.dirname(__file__)
data_dir   = os.path.join(script_dir, f"n{nlayers}_q{nqubits}_u{ulayers}_threads_bench_matchgate")


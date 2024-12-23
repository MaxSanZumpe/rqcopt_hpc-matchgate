#include <assert.h>
#include <complex.h>
#include "statevector.h"
#include "matchgate_brickwall.h"
#include "gate.h"
#include "matchgate.h"
#include "timing.h"
#include "util.h"


int main()
{   
    const int nlayers = 3;
    const int nqubits = 12;
    const int ulayers = 601;

    const intqs n = (intqs)1 << nqubits;
    struct statevector psi0, Upsi;
    if (allocate_statevector(nqubits, &psi0) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Upsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }

    char filename[1024];
    sprintf(filename, "../benchmark_hpc/bench_in/q%i/n%i_q%i_u%i_bench_in.hdf5", nqubits, nlayers, nqubits, ulayers);
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    char varname[32];
    struct matchgate ulist[ulayers];
    if (read_hdf5_dataset(file, "ulist", H5T_NATIVE_DOUBLE, ulist) < 0) {
        fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
        return -1;
    }

    int uperms[ulayers][nqubits];
    for (int i = 0; i < ulayers; i++)
    {
        sprintf(varname, "uperm%i", i);
        if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
            fprintf(stderr, "reading permutation data from disk failed\n");
            return -1;
        }
    }


    if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi0.data) < 0) {
        fprintf(stderr, "reading input statevector data from disk failed");
        return -1;
    }

    H5Fclose(file);

    const int* upperms[ulayers];
    for (int l = 0; l < ulayers; l++) {
        upperms[l] = uperms[l];
    }


    uint64_t start_tick = get_ticks();
    apply_matchgate_brickwall_unitary(ulist, ulayers, upperms, &psi0, &Upsi);
    uint64_t total_ticks = get_ticks() - start_tick;

    // get the tick resolution
    const double ticks_per_sec = (double)get_tick_resolution();
    double wtime = (double) total_ticks / ticks_per_sec;
        
    sprintf(filename, "../benchmark_hpc/bench_out/q%i/usplit/q%i_u%i_usplit_bench.hdf5", nqubits, nqubits, ulayers);
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}
    
	if (write_hdf5_scalar_attribute(file, "Walltime", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &wtime)) {
		fprintf(stderr, "writing 'Walltime' to disk failed\n");
		return -1;
	}

    if (write_hdf5_scalar_attribute(file, "ulayers", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &ulayers)) {
		fprintf(stderr, "writing 'ulayers' to disk failed\n");
		return -1;
	}

    H5Fclose(file);

    free_statevector(&psi0);
    free_statevector(&Upsi);
}

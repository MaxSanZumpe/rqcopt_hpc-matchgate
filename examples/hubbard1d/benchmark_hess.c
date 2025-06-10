#include <omp.h>
#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include "mg_brickwall_opt.h"
#include "util.h"
#include "timing.h"
#include "matchgate_brickwall.h"


struct u_splitting
{
	struct matchgate* ulist;
	const int   ulayers;
	const int** upperms;
};

static int ufunc_matfree(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out)
{
	const struct u_splitting* U = fdata;

	// apply U in brickwall form

	apply_matchgate_brickwall_unitary(U->ulist, U->ulayers, U->upperms, psi, psi_out);
	
	return 0;
}



int main()
{
	const int nqubits = 16;

    const int nlayers = 5;
    const int ulayers = 81;

    const int order = 2;
    const float t = 0.2;
    const float g = 4.0;

	// read initial data from disk
	char filename[1024];
	sprintf(filename, "../examples/hubbard1d/input/hubbard1d_suzuki%i_n%i_q%i_u%i_t%.2fs_g%.2f_init.hdf5", order, nlayers, nqubits, ulayers, t, g);
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

	int nlayers_ref;
	if (read_hdf5_attribute(file, "nlayers", H5T_NATIVE_INT, &nlayers_ref) < 0) {
		fprintf(stderr, "reading 'nlayers' from disk failed\n");
		return -1;
	}

	int nqubits_ref;
	if (read_hdf5_attribute(file, "nqubits", H5T_NATIVE_INT, &nqubits_ref) < 0) {
		fprintf(stderr, "reading 'nqubits_ref' from disk failed\n");
		return -1;
	}

	assert(nqubits == nqubits_ref);
	assert(nlayers == nlayers_ref);


	struct matchgate* ulist = aligned_alloc(MEM_DATA_ALIGN, ulayers * sizeof(struct matchgate));
	if (read_hdf5_dataset(file, "ulist", H5T_NATIVE_DOUBLE, (numeric*)ulist) < 0) {
		fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
		return -1;
	}

	int uperms[ulayers][nqubits];
	for (int i = 0; i < ulayers; i++)
	{
		char varname[32];
		sprintf(varname, "uperm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed\n");
			return -1;
		}
	}
	const int* upperms[ulayers];

	for (int i = 0; i < ulayers; i++){
		upperms[i] = uperms[i];
	}
	
	struct u_splitting udata = {
		.ulist   = ulist,
		.ulayers = ulayers,
		.upperms = upperms,
	};


	// initial to-be optimized quantum gates
	struct matchgate* vlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, (numeric*)vlist) < 0) {
		fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
		return -1;
	}

	// permutations
	int perms[nlayers][nqubits];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed\n");
			return -1;
		}
	}
	const int* pperms[nlayers];

	for (int i = 0; i < nlayers; i++){
		pperms[i] = perms[i];
	}

    struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		fprintf(stderr, "reading input statevector data from disk failed");
		return -1;
	}

	H5Fclose(file);

	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(nqubits, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache = { 0 };
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

    struct matchgate* dvlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (dvlist == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));
	}


    const int m = nlayers * 8;
    numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	memset(hess, 0, m * m * sizeof(numeric));

	memset(psi.data, 0, n * sizeof(numeric));
	psi.data[b] = 1;

    int ret = ufunc_matfree(&psi, &udata, &Upsi);
    if (ret < 0) {
        fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
        return -2;
    }

    const intqs n = (intqs)1 << nqubits;
    // negate and complex-conjugate entries
    for (intqs a = 0; a < n; a++)
    {
        Upsi.data[a] = -conj(Upsi.data[a]);
    }
	
	matchgate_brickwall_unitary_forward(vlist, nlayers, pperms, &psi, &cache, &Wpsi);

	uint64_t start_tick = get_ticks();
	matchgate_brickwall_unitary_backward_hessian(vlist, nlayers, pperms, &cache, &Upsi, &psi, dvlist, hess);
	uint64_t total_ticks = get_ticks() - start_tick;

    // get the tick resolution
    const double ticks_per_sec = (double)get_tick_resolution();
    const double wtime0 = (double) total_ticks / ticks_per_sec;

	
	start_tick = get_ticks();
	
	matchgate_brickwall_unitary_forward(vlist, nlayers, pperms, &psi, &cache, &Wpsi);
	matchgate_ti_brickwall_unitary_backward_hessian(vlist, nlayers, pperms, &cache, &Upsi, &psi, dvlist, hess);
	
	total_ticks = get_ticks() - start_tick;
    const double wtime1 = (double) total_ticks / ticks_per_sec;


	printf("q = %i : w0 = %lf | w1 = %lf\n", nqubits, wtime0, wtime1); 

    // save results to disk
    sprintf(filename, "../examples/hubbard1d/bench_data/hubbard1d_hess_bench_n%i_q%i_u%i.hdf5", nlayers, nqubits, ulayers);
    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    // store parameters
    if (write_hdf5_scalar_attribute(file, "nqubits", H5T_STD_I32LE, H5T_NATIVE_INT, &nqubits)) {
        fprintf(stderr, "writing 'nqubits' to disk failed\n");
        return -1;
    }

    if (write_hdf5_scalar_attribute(file, "nlayers", H5T_STD_I32LE, H5T_NATIVE_INT, &nlayers)) {
        fprintf(stderr, "writing 'nqubits' to disk failed\n");
        return -1;
    }

    if (write_hdf5_scalar_attribute(file, "ulayers", H5T_STD_I32LE, H5T_NATIVE_INT, &ulayers)) {
        fprintf(stderr, "writing 'nqubits' to disk failed\n");
        return -1;
    }

    // store run-time diagnostics
    if (write_hdf5_scalar_attribute(file, "Walltime_inv0", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &wtime0)) {
        fprintf(stderr, "writing 'Walltime' to disk failed\n");
        return -1;
    }

	if (write_hdf5_scalar_attribute(file, "Walltime_inv1", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &wtime1)) {
        fprintf(stderr, "writing 'Walltime' to disk failed\n");
        return -1;
    }

    H5Fclose(file);

	

	free_statevector(&psi);
	free_statevector(&Upsi);
	free_statevector(&Wpsi);

	free_quantum_circuit_cache(&cache);

	aligned_free(vlist);
	aligned_free(ulist);
	aligned_free(hess);
	aligned_free(dvlist);

	return 0;
};

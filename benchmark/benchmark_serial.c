#include <omp.h>
#include <stdio.h>
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
	#ifdef STATEVECTOR_PARALLELIZATION

    printf("Please recompile with parallelization disabled\n."); 
    return 0;

    #else

	const int nqubits = 8;
	const int nlayers = 4;
	int ulayers;

    int num_threads = 1;

	// read initial data from disk
	char filename[1024];
	sprintf(filename, "/dss/dsshome1/01/ge47gej2/rqcopt_hpc/examples/benchmark/input_data/spinless_hubbard_n%i_q%i_matchgate_init.hdf5", nlayers, nqubits);
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

	int nlayers_ref;
	if (read_hdf5_dataset(file, "nlayers", H5T_NATIVE_INT, &nlayers_ref) < 0) {
		fprintf(stderr, "reading 'nlayers' from disk failed\n");
		return -1;
	}

	int nqubits_ref;
	if (read_hdf5_dataset(file, "nqubits", H5T_NATIVE_INT, &nqubits_ref) < 0) {
		fprintf(stderr, "reading 'nqubits_ref' from disk failed\n");
		return -1;
	}

	assert(nqubits == nqubits_ref);
	assert(nlayers == nlayers_ref);

	if (read_hdf5_dataset(file, "ulayers", H5T_NATIVE_INT, &ulayers) < 0) {
		fprintf(stderr, "reading 'ulayers' from disk failed\n");
		return -1;
	}

	struct matchgate* u_split = aligned_alloc(MEM_DATA_ALIGN, ulayers * sizeof(struct matchgate));
	if (read_hdf5_dataset(file, "Ulist", H5T_NATIVE_DOUBLE, (numeric*)u_split) < 0) {
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
		.ulist   = u_split,
		.ulayers = ulayers,
		.upperms = upperms,
	};

	// initial to-be optimized quantum gates
	struct matchgate* vlist_start = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (read_hdf5_dataset(file, "Vlist_start", H5T_NATIVE_DOUBLE, (numeric*)vlist_start) < 0) {
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

	H5Fclose(file);


	// parameters for optimization
	struct rtr_params params;
	set_rtr_default_params(nlayers * 16, &params);

	// number of iterations
	const int niter = 1;
	
	double* f_iter = aligned_alloc(MEM_DATA_ALIGN, (niter + 1) * sizeof(double));

	struct matchgate* vlist_opt = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	
    uint64_t start_tick = get_ticks();
    
    // perform optimization
    optimize_matchgate_brickwall_circuit_hmat(ufunc_matfree, &udata, vlist_start, nlayers, nqubits, pperms, &params, niter, f_iter, vlist_opt);

    uint64_t total_ticks = get_ticks() - start_tick;
    // get the tick resolution
    const double ticks_per_sec = (double)get_tick_resolution();
    const double wtime = (double) total_ticks / ticks_per_sec;


    int translational_invariance = 0;
    int statevector_parallelization = 0;
    int gate_parallelization = 0;

    #ifdef TRANSLATIONAL_INVARIANCE
    translational_invariance = 1;
    #endif

    #ifdef STATEVECTOR_PARALLELIZATION
    statevector_parallelization = 1;
    #endif

    #ifdef GATE_PARALLELIZATION
    gate_parallelization = 1;
    #endif

    // save results to disk
    sprintf(filename, "../examples/benchmark/output_data/spinless_hubbard_n%i_q%i_th%i_%i%i%i.hdf5", nlayers, nqubits, num_threads, translational_invariance, statevector_parallelization, gate_parallelization);
    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    hsize_t vdims[3] = { nlayers, 8, 2 };
    if (write_hdf5_dataset(file, "vlist_opt", 3 , vdims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, (numeric*)vlist_opt) < 0) {
        fprintf(stderr, "writing 'vlist_opt' to disk failed\n");
        return -1;
    }

    hsize_t fdims[1] = { niter + 1 };
    if (write_hdf5_dataset(file, "f_iter", 1, fdims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, f_iter) < 0) {
        fprintf(stderr, "writing 'f_iter' to disk failed\n");
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
    if (write_hdf5_scalar_attribute(file, "Walltime", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &wtime)) {
        fprintf(stderr, "writing 'Walltime' to disk failed\n");
        return -1;
    }


    if (write_hdf5_scalar_attribute(file, "TRANSLATIONAL_INVARIANCE", H5T_STD_I32LE, H5T_NATIVE_INT, &translational_invariance)) {
        fprintf(stderr, "writing 'TRANSLATIONAL_INVARIANCE' to disk failed\n");
        return -1;
    }

    if (write_hdf5_scalar_attribute(file, "STATEVECTOR_PARALLELIZATION", H5T_STD_I32LE, H5T_NATIVE_INT, &statevector_parallelization )) {
        fprintf(stderr, "writing 'STATEVECTOR_PARALLELIZATION' to disk failed\n");
        return -1;
    }

    if (write_hdf5_scalar_attribute(file, "GATE_PARALLELIZATION", H5T_STD_I32LE, H5T_NATIVE_INT, &gate_parallelization )) {
        fprintf(stderr, "writing 'STATEVECTOR_PARALLELIZATION' to disk failed\n");
        return -1;
    }

    if (write_hdf5_scalar_attribute(file, "NUM_THREADS", H5T_STD_I32LE, H5T_NATIVE_INT, &num_threads)) {
        fprintf(stderr, "writing 'NUM_THREADS_STATEVECTOR_PARALLELIZATION' to disk failed\n");
        return -1;
    }

    H5Fclose(file);

	

	aligned_free(vlist_opt);
	aligned_free(f_iter);
	aligned_free(vlist_start);
	aligned_free(u_split);

    #endif

	return 0;
};

#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include "mg_brickwall_opt.h"
#include "util.h"
#include "timing.h"
#include "matchgate_brickwall.h"

#ifdef LRZ_HPC
	#include <mkl_cblas.h>
#else
	#include <cblas.h>
#endif

int get_num_threads(void) {
    int num_threads = 0;
    #pragma omp parallel reduction(+:num_threads)
    num_threads += 1;
    return num_threads;
}


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

static int ufunc(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out)
{
	const intqs n = (intqs)1 << psi->nqubits;
	const numeric* U = (numeric*)fdata;

	// apply U
	numeric alpha = 1;
	numeric beta  = 0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &alpha, U, n, psi->data, 1, &beta, psi_out->data, 1);

	return 0;
}



int main()
{
	const int nqubits = 8;
	const int nlayers = 81;
	
	const int full_target = 1;
	const int ulayers = 29;
	
	if (full_target == 1) { assert(ulayers == 29); }

	char splitting[] = "suzuki4";

	float g = 1.50;
    float t = 0.25;

	const int niter = 150;

	int num_threads;
	#if  defined(STATEVECTOR_PARALLELIZATION) || defined(GATE_PARALLELIZATION)
	num_threads = get_num_threads();
	#else 
	num_threads = 1;
	   #endif

	// read initial data from disk
	char filename[1024];
	
	numeric* expiH;
	if (full_target == 1) {
		sprintf(filename, "../examples/hubbard1d/opt_in/q%i/hubbard1d_q%i_unitary_t%.2fs_g%.2f_init.hdf5", nqubits, nqubits, t, g);
		hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

		const intqs n = (intqs)1 << nqubits;

		expiH = aligned_alloc(MEM_DATA_ALIGN, n * n * sizeof(numeric));
		if (expiH == NULL) {
			fprintf(stderr, "memory allocation for target unitary failed\n");
			return -1;
		}
		if (read_hdf5_dataset(file, "expiH", H5T_NATIVE_DOUBLE, expiH) < 0) {
			fprintf(stderr, "reading 'expiH' from disk failed\n");
			return -1;
		}
	
		H5Fclose(file);
	}


	sprintf(filename, "../examples/hubbard1d/opt_in/q%i/hubbard1d_%s_n%i_q%i_u%i_t%.2fs_g%.2f_init.hdf5", nqubits, splitting, nlayers, nqubits, ulayers, t, g);
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

	struct matchgate* ulist = aligned_alloc(MEM_DATA_ALIGN, ulayers * sizeof(struct matchgate));
	if (read_hdf5_dataset(file, "ulist", H5T_NATIVE_DOUBLE, (numeric*)ulist) < 0) {
		fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
		return -1;
	} 


	int uperms[ulayers][nqubits];
	for (int i = 0; i < ulayers; i++) {
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

	struct u_splitting udata_split = {
		.ulist   = ulist,
		.ulayers = ulayers,
		.upperms = upperms,
	};


	// initial to-be optimized quantum gates
	struct matchgate* vlist_start = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, (numeric*)vlist_start) < 0) {
		fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
		return -1;
	}

	// permutations
	int perms[nlayers][nqubits];
	for (int i = 0; i < nlayers; i++) {
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed\n");
			return -1;
		}
	}

	const int* pperms[nlayers];
	for (int i = 0; i < nlayers; i++) {
		pperms[i] = perms[i];
	}
	H5Fclose(file);


	// parameters for optimization
	struct rtr_params params;
	set_rtr_default_params(nlayers * 16, &params);

	// target function
	
	double* f_iter = aligned_alloc(MEM_DATA_ALIGN, (niter + 1) * sizeof(double));

	struct matchgate* vlist_opt = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));

	void* udata;

	linear_func func;
	if (full_target == 1) {
		func = ufunc;
		udata = expiH;

	} else {
		func = ufunc_matfree;
		udata = &udata_split;
	}	
	
	uint64_t start_tick = get_ticks();
	
	// perform optimization
	optimize_matchgate_brickwall_circuit_hmat(func, udata, vlist_start, nlayers, nqubits, pperms, &params, niter, f_iter, vlist_opt);

	uint64_t total_ticks = get_ticks() - start_tick;
	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();
	const double wtime = (double) total_ticks / ticks_per_sec;

	const intqs m = (intqs)1 << nqubits;
	double norm_start = sqrt(2*f_iter[0    ] + 2*m);
	double norm_final = sqrt(2*f_iter[niter] + 2*m);
	printf("\nStart norm (frobenius) error: %f", norm_start);
	printf("\nFinal norm (frobenius) error: %f\n", norm_final);

	int translational_invariance = 0;
	int statevector_parallelization = 0;
	int gate_parallelization = 0;

	int hpc = 0;
	int mpi = 0;

	#ifdef TRANSLATIONAL_INVARIANCE
	translational_invariance = 1;
	#endif

	#ifdef STATEVECTOR_PARALLELIZATION
	statevector_parallelization = 1;
	#endif

	#ifdef GATE_PARALLELIZATION
	gate_parallelization = 1;
	#endif

	#ifdef LRZ_HPC
	hpc = 1;
	#endif

	#ifdef MPI
	mpi = 1;
	#endif


	// save results to disk
	int temp;
	if (full_target == 1) {
		temp = 0;
	} else {
		temp = ulayers;
	}
	sprintf(filename, "../examples/hubbard1d/opt_out/q%i/hubbard1d_%s_n%i_q%i_u%i_t%.2fs_g%.2f_iter%i_inv%i_opt.hdf5", nqubits, splitting, nlayers, nqubits, temp, t, g, niter, translational_invariance);
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

	// store run-time diagnostics
	if (write_hdf5_scalar_attribute(file, "Walltime", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &wtime)) {
		fprintf(stderr, "writing 'Walltime' to disk failed\n");
		return -1;
	}

	if (write_hdf5_scalar_attribute(file, "FULL_TARGET", H5T_STD_I32LE, H5T_NATIVE_INT, &full_target)) {
		fprintf(stderr, "writing 'NUM_THREADS_STATEVECTOR_PARALLELIZATION' to disk failed\n");
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

	if (write_hdf5_scalar_attribute(file, "LRZ_HPC", H5T_STD_I32LE, H5T_NATIVE_INT, &hpc)) {
		fprintf(stderr, "writing 'NUM_THREADS_STATEVECTOR_PARALLELIZATION' to disk failed\n");
		return -1;
	}

	if (write_hdf5_scalar_attribute(file, "MPI", H5T_STD_I32LE, H5T_NATIVE_INT, &mpi)) {
		fprintf(stderr, "writing 'NUM_THREADS_STATEVECTOR_PARALLELIZATION' to disk failed\n");
		return -1;
	}

	H5Fclose(file);

	aligned_free(vlist_opt);
	aligned_free(f_iter);
	aligned_free(vlist_start);
	aligned_free(ulist);

	return 0;
};

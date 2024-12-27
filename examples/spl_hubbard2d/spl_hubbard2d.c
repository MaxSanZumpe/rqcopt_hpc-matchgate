#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mg_brickwall_opt.h"
#include "util.h"
#include "timing.h"


struct u_splitting
{
	struct matchgate* ulist;
	const int   ulayers;
	const int** upperms;
};

int get_num_threads() {
    int num_threads = 0;
    #pragma omp parallel reduction(+:num_threads)
	{
    	num_threads += 1;
	}
    return num_threads;
}

void asign_state_vectors(int nqubits, int rank, int num_tasks, int* start, int* end)
{
    intqs dim = (long int)1 << nqubits;
    
    int div = dim / num_tasks;
	int res = dim % num_tasks;

	int s, e;

	if(rank < res) {
    	s = rank * (div + 1);
    	e = rank * (div + 1) + div + 1;
	} else {
		s = (rank - res) * div + res * (div + 1);
    	e = (rank - res) * div + res * (div + 1) + div;
	}

	*start = s;
	*end = e;
}

static int ufunc_matfree(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out)
{
	const struct u_splitting* U = udata;

	// apply U in brickwall form

	apply_matchgate_brickwall_unitary(U->ulist, U->ulayers, U->upperms, psi, psi_out);
	
	return 0;
}


int main()
{	
    MPI_Init(NULL, NULL);

    int rank;
    int num_tasks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const int nqubits = 16;
	const int nlayers = 31;
	const int ulayers = 601;
	const int order = 4;

    char model[] = "suzuki";

    float g = 1.50;
    float t = 0.25;


	int num_threads;
	#if  defined(STATEVECTOR_PARALLELIZATION) || defined(GATE_PARALLELIZATION)
	num_threads = get_num_threads();
	#else 
	num_threads = 1;
	#endif


	// declaring variables here to mantain shared scope

	int perms[nlayers][nqubits];
	const int* pperms[nlayers];

	int uperms[ulayers][nqubits];
	const int* upperms[ulayers];

	struct matchgate* u_split = aligned_alloc(MEM_DATA_ALIGN, ulayers * sizeof(struct matchgate));

	struct matchgate* vlist_start = NULL;
	struct matchgate* vlist_opt = NULL;

	const int niter = 10;
	struct rtr_params params;	
	double* f_iter = NULL;

	// we want to read the input data only on one node.	
	if(rank == 0)
	{ 	
		char filename[1024];
        sprintf(filename, "../examples/spl_hubbard2d/opt_in/spl_hubbard2d_%s%i_n%i_q%i_u%i_t%.2fs_g%.2f_init.hdf5", model , order, nlayers, nqubits, ulayers, t, g);


		hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file < 0) {
			fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		int nlayers_ref;
		if (read_hdf5_dataset(file, "nlayers", H5T_NATIVE_INT, &nlayers_ref) < 0) {
			fprintf(stderr, "reading 'nlayers' from disk failed\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		int nqubits_ref;
		if (read_hdf5_dataset(file, "nqubits", H5T_NATIVE_INT, &nqubits_ref) < 0) {
			fprintf(stderr, "reading 'nqubits_ref' from disk failed\n");
			return -1;
		}

		int ulayers_ref;
		if (read_hdf5_dataset(file, "ulayers", H5T_NATIVE_INT, &ulayers_ref) < 0) {
			fprintf(stderr, "reading 'ulayers' from disk failed\n");
			return -1;
		}

		assert(nqubits == nqubits_ref);
		assert(nlayers == nlayers_ref);
		assert(ulayers == ulayers_ref);

		if (read_hdf5_dataset(file, "ulist", H5T_NATIVE_DOUBLE, (numeric*)u_split) < 0) {
			fprintf(stderr, "reading target unitary two qubit splitting gates failed.\n");
			return -1;
		}

		for (int i = 0; i < ulayers; i++)
		{
			char varname[32];
			sprintf(varname, "uperm%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
				fprintf(stderr, "reading permutation data from disk failed\n");
				return -1;
			}
		}

		// initial to-be optimized quantum gates
		vlist_start = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
		if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, (numeric*)vlist_start) < 0) {
			fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
			return -1;
		}

		for (int i = 0; i < nlayers; i++)
		{
			char varname[32];
			sprintf(varname, "perm%i", i);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
				fprintf(stderr, "reading permutation data from disk failed\n");
				return -1;
			}
		}
		H5Fclose(file);

		// parameters for optimization
		set_rtr_default_params(nlayers * 16, &params);

		f_iter = aligned_alloc(MEM_DATA_ALIGN, (niter + 1) * sizeof(double));

		vlist_opt = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	}	

	MPI_Bcast((void*)u_split, 8*ulayers, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	MPI_Bcast((void*)uperms, ulayers * nqubits, MPI_INT, 0, MPI_COMM_WORLD);


	for (int i = 0; i < ulayers; i++){
		upperms[i] = uperms[i];
	}

	struct u_splitting udata = {
		.ulist   = u_split,
		.ulayers = ulayers,
		.upperms = upperms,
	};


	MPI_Bcast((void*)perms, nlayers * nqubits, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < nlayers; i++){
		pperms[i] = perms[i];
	}
	
	// perform optimization
	int start, end;
	asign_state_vectors(nqubits, rank, num_tasks, &start, &end);

	uint64_t start_tick = get_ticks();
	mpi_optimize_matchgate_brickwall_circuit_hmat(rank, start, end, ufunc_matfree, &udata, vlist_start, nlayers, nqubits, pperms, &params, niter, f_iter, vlist_opt);
	uint64_t total_ticks = get_ticks() - start_tick;

	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();
	const double wtime = (double) total_ticks / ticks_per_sec;

	if (rank == 0)
	{
		int translational_invariance = 0;
		int statevector_parallelization = 0;

		#ifdef TRANSLATIONAL_INVARIANCE
		translational_invariance = 1;
		#endif

		#ifdef STATEVECTOR_PARALLELIZATION
		statevector_parallelization = 1;
		#endif

		char filename[1024];
        	sprintf(filename, "../examples/spl_hubbard2d/opt_out/spl_hubbard2d_%s%i_n%i_q%i_u%i_t%.2fs_g%.2f_opt_iter%i.hdf5", model, order, nlayers, nqubits, ulayers, t, g, niter);


		hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
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

		if (write_hdf5_scalar_attribute(file, "NUM_THREADS", H5T_STD_I32LE, H5T_NATIVE_INT, &num_threads)) {
			fprintf(stderr, "writing 'NUM_THREADS_STATEVECTOR_PARALLELIZATION' to disk failed\n");
			return -1;
		}

		if (write_hdf5_scalar_attribute(file, "NUM_TASKS", H5T_STD_I32LE, H5T_NATIVE_INT, &num_tasks)) {
			fprintf(stderr, "writing 'NUM_TASKS' to disk failed\n");
			return -1;
		}

		H5Fclose(file);
	}

	if (rank == 0)
	{
		aligned_free(vlist_opt);
		aligned_free(f_iter);
		aligned_free(vlist_start);
	}

	aligned_free(u_split);

	MPI_Finalize();

	return 0;
};

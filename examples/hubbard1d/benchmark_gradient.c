#include <stdio.h>
#include <assert.h>
#include "util.h"
#include "timing.h"
#include "matchgate_brickwall.h"
#include "matchgate_target.h"


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
    const int nqubits = 12;

    const int nlayers = 5;
    const int ulayers = 309;

    const int order = 2;
    const float t = 0.75;
    const float g = 4.0;

    numeric f;

    struct matchgate* vlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
    struct matchgate* ulist = aligned_alloc(MEM_DATA_ALIGN, ulayers * sizeof(struct matchgate));

    if (vlist == NULL || ulist == NULL) {
		fprintf(stderr, "memory allocation for gates failed.\n");
		return -1;
	}

	struct matchgate* dvlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (dvlist == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed.\n", nlayers);
		return -1;
	}

    char filename[1024];
    sprintf(filename, "../examples/hubbard1d/input/hubbard1d_suzuki%i_n%i_q%i_u%i_t%.2fs_g%.2f_init.hdf5", order, nlayers, nqubits, ulayers, t, g);
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    char varname[32];

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

    int uperms[ulayers][nqubits];
    for (int i = 0; i < ulayers; i++)
    {
        sprintf(varname, "uperm%i", i);
        if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
            fprintf(stderr, "reading permutation data from disk failed\n");
            return -1;
        }
    }

    H5Fclose(file);

    const int* pperms[nlayers];
    for (int i = 0; i < nlayers; i++){
        pperms[i] = perms[i];
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

    uint64_t start_tick = get_ticks();

    matchgate_brickwall_unitary_target_and_gradient(ufunc_matfree, &udata, vlist, nlayers, nqubits, pperms, &f, dvlist);

    uint64_t total_ticks = get_ticks() - start_tick;

    // get the tick resolution
    const double ticks_per_sec = (double)get_tick_resolution();
    double wtime = (double) total_ticks / ticks_per_sec;
    printf("wtime: %lf\n", wtime);

    sprintf(filename, "../examples/hubbard1d/bench_data/hubbard1d_suzuki%i_n%i_q%i_u%i_t%.2fs_g%.2f_grad.hdf5", order, nlayers, nqubits, ulayers, t, g);
    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    if (write_hdf5_scalar_attribute(file, "nqubits", H5T_STD_I32LE, H5T_NATIVE_INT, &nqubits)) {
		fprintf(stderr, "writing 'nqubits' to disk failed\n");
		return -1;
	}

	if (write_hdf5_scalar_attribute(file, "nlayers", H5T_STD_I32LE, H5T_NATIVE_INT, &nlayers)) {
		fprintf(stderr, "writing 'nqubits' to disk failed\n");
		return -1;
	}

    if (write_hdf5_scalar_attribute(file, "Walltime", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &wtime)) {
		fprintf(stderr, "writing 'Walltime' to disk failed\n");
		return -1;
	}

    H5Fclose(file);

    free(vlist);
    free(ulist);
    free(dvlist);
}   


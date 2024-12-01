#include <stdio.h>
#include <assert.h>
#include "util.h"
#include "matchgate_brickwall.h"
#include "statevector.h"


int main() 
{
    const int nqubits = 8;
    const int nlayers = 8;

    const int ulayers = 81;
    const int order   = 2;

    struct matchgate* ulist = aligned_alloc(MEM_DATA_ALIGN, ulayers * sizeof(struct matchgate));
    struct matchgate* vlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));

    struct statevector Spsi;
    if (allocate_statevector(nqubits, &Spsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }

    char filename[1024];
    sprintf(filename, "../examples/spl_hubbard1d/input/spl_hubbard1d_suzuki%i_n%i_q%i_u%i_t0.25s_init.hdf5", order, nlayers, nqubits, ulayers);
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    if (read_hdf5_dataset(file, "ulist", H5T_NATIVE_DOUBLE, ulist) < 0) {
        fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
        return -1;
    }

    char varname[32];
    int uperms[ulayers][nqubits];
	for (int i = 0; i < ulayers; i++)
	{
		sprintf(varname, "uperm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed\n");
			return -1;
		}
	}

    const int* upperms[ulayers];
    for (int l = 0; l < ulayers; l++) {
	    upperms[l] = uperms[l];
    }

    int ulayers_ref = 0;
    if (read_hdf5_attribute(file, "ulayers", H5T_NATIVE_INT, &ulayers_ref) < 0) {
        fprintf(stderr, "reading 'ulayers' from disk failed\n");
        return -1;
    }

    assert(ulayers == ulayers_ref);

    if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, Spsi.data) < 0) {
        fprintf(stderr, "reading input statevector data from disk failed");
        return -1;
    }

    struct matchgate* vlist_start = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
    if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, vlist) < 0) {
        fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
        return -1;
    }


    H5Fclose(file);


    sprintf(filename, "../examples/spl_hubbard1d/input/hubbard1d_opt_n%i_q%i_th8.hdf5", nlayers, nqubits);
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    if (read_hdf5_dataset(file, "Vlist_start", H5T_NATIVE_DOUBLE, vlist) < 0) {
        fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
        return -1;
    }

    
    int perms[nlayers][nqubits];
	for (int i = 0; i < nlayers; i++)
	{
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed\n");
			return -1;
		}
	}

    const int* pperms[nlayers];
    for (int k = 0; k < nlayers; k++) {
	    pperms[k] = perms[k];
    }

    H5Fclose(file);

    aligned_free(ulist);
    aligned_free(vlist);
    free_statevector(&Spsi);
    return 0;
}


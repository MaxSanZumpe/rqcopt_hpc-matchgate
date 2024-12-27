#include <assert.h>
#include <complex.h>
#include "statevector.h"
#include "matchgate_brickwall.h"
#include "gate.h"
#include "matchgate.h"
#include "util.h"


int main()
{   
    const int nqubits = 16;
    const int nlayers = 31;
    const int steps = 5;

    char model[] = "suzuki2";

    const int ulayers = 601;

    const int r = 6;

    float g = 1.50;
    float t = 0.25;

    const int niter = 10;

    const intqs n = (intqs)1 << nqubits;
    struct statevector psi0, Upsi, Upsi_ref, diff;
    if (allocate_statevector(nqubits, &psi0) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Upsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Upsi_ref) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &diff) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }

    double errors[steps];
    int layers_arr[steps];

    hid_t file;
    char filename[1024];

    sprintf(filename, "../examples/spl_hubbard2d/test_splitting/error_in/spl_hubbard2d_suzuki2_q16_us150_u901_t%.2fs_g%.2f_error_in.hdf5", model, nqubits, t, g);
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
        return -1;
    }

    if (read_hdf5_dataset(file, "psi0", H5T_NATIVE_DOUBLE, psi0.data) < 0) {
        fprintf(stderr, "reading input statevector data from disk failed");
        return -1;
    }

    if (read_hdf5_dataset(file, "Upsi", H5T_NATIVE_DOUBLE, Upsi_ref.data) < 0) {
        fprintf(stderr, "reading input statevector data from disk failed");
        return -1;
    }

    H5Fclose(file);

    for (int s = 1; s <= (nlayers-1)/r ; s++) {
        
        int l = r*s + 1;
        sprintf(filename, "../examples/spl_hubbard2d/opt_out/spl_hubbard2d_%s_n%i_q%i_u%i_t%.2fs_g%.2f_opt_iter%i.hdf5", nqubits, model, l, nqubits, ulayers, t, g, niter);
        file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
            return -1;
        }

        struct matchgate vlist[l];
        char varname[32];
        if (read_hdf5_dataset(file, "vlist_opt", H5T_NATIVE_DOUBLE, vlist) < 0) {
            fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
            return -1;
        }

        int layers_ref;
        read_hdf5_attribute(file, "nlayers", H5T_NATIVE_INT, &layers_ref);

        assert(l == layers_ref);   

        H5Fclose(file);

        sprintf(filename, "../examples/spl_hubbard2d/opt_in/spl_hubbard2d_%s_n%i_q%i_u%i_t%.2fs_g%.2f_init.hdf5", nqubits, model, l, nqubits, ulayers, t, g);
        file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
            return -1;
        }

        int perms[l][nqubits];
        for (int i = 0; i < l; i++)
        {
            sprintf(varname, "perm%i", i);
            if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
                fprintf(stderr, "reading permutation data from disk failed\n");
                return -1;
            }
        }

        H5Fclose(file);

        const int* pperms[l];
        for (int j = 0; j < l; j++) {
            pperms[j] = perms[j];
        }

        apply_matchgate_brickwall_unitary(vlist, l, pperms, &psi0, &Upsi);

        double norm_error = uniform_distance(n, Upsi_ref.data, Upsi.data);

        printf("layers: %i | %lf\n", l, norm_error);
        errors[s - 1] = norm_error;
        layers_arr[s - 1] = l;
    }

    sprintf(filename, "../examples/spl_hubbard2d/test_splitting/error_out/spl_hubbard2d_%s_opt_n%i_q%i_t%.2fs_g%.2f_iter%i.hdf5", model, nlayers, nqubits, t, g, niter);
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

    
	hsize_t dims[1] = { steps };
	if (write_hdf5_dataset(file, "errors", 1, dims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, errors) < 0) {
		fprintf(stderr, "writing 'f_iter' to disk failed\n");
		return -1;
	}

	if (write_hdf5_dataset(file, "layers", 1, dims, H5T_STD_I32LE, H5T_NATIVE_INT, layers_arr) < 0) {
		fprintf(stderr, "writing 'f_iter' to disk failed\n");
		return -1;
	}

    H5Fclose(file);

    free_statevector(&psi0);
    free_statevector(&Upsi);
    free_statevector(&Upsi_ref);
    free_statevector(&diff);

}

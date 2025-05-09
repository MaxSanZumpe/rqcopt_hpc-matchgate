#include <assert.h>
#include <complex.h>
#include "statevector.h"
#include "matchgate_brickwall.h"
#include "gate.h"
#include "matchgate.h"
#include "util.h"


int main()
{   
    const int nqubits = 12;
    const int nlayers = 41;
    const int s_start = 2;
    const int steps = 1;

    char model[] = "suzuki4";

    const int ulayers = 1009;

    const int r = 20;

    float g = 4.00;
    float t = 1.00;

    const int niter = 15;

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

    sprintf(filename, "../examples/hubbard1d/test_splitting/error_in/hubbard1d_suzuki2_q%i_us200_u801_t%.2fs_g%.2f_error_in.hdf5", nqubits, t, g);
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

    for (int s = s_start; s <= (nlayers-1)/r ; s++) {
        
        int l = r*s + 1;
        sprintf(filename, "../examples/hubbard1d/opt_out/q%i/hubbard1d_%s_n%i_q%i_u%i_t%.2fs_g%.2f_iter%i_opt.hdf5", nqubits, model, l, nqubits, ulayers, t, g, niter);
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

        sprintf(filename, "../examples/hubbard1d/opt_in/q%i/hubbard1d_%s_n%i_q%i_u%i_t%.2fs_g%.2f_init.hdf5", nqubits, model, l, nqubits, ulayers, t, g);
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
        errors[s - s_start] = norm_error;
        layers_arr[s - s_start] = l;
    }

    sprintf(filename, "../examples/hubbard1d/test_splitting/error_out/hubbard1d_%s_opt_n%i_q%i_t%.2fs_g%.2f_iter%i.hdf5", model, nlayers, nqubits, t, g, niter);
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

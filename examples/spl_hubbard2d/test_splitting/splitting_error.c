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

    char model[] = "suzuki6";

    const int us = 4;
    const int ulayers = 601;

    const int r = 150;

    float g = 1.5;
    float t = 0.25;

    const intqs n = (intqs)1 << nqubits;
    struct statevector psi0, Upsi, Upsi_ref, diff;
    if (allocate_statevector(nqubits, &psi0) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Upsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Upsi_ref) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &diff) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }

    double errors[us];
    int layers_arr[us];

    hid_t file;
    char filename[1024];
    for (int s = 1; s <= us; s++) {
        
        int layers = r*s + 1;
        sprintf(filename, "../examples/spl_hubbard2d/test_splitting/error_in/spl_hubbard2d_%s_q%i_us%i_u%i_t%.2fs_g%.2f_error_in.hdf5", model, nqubits, us, ulayers, t, g);
        file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
            return -1;
        }

        struct matchgate ulist[layers];
        char varname[32];
        sprintf(varname, "ulist_%i", s);
        if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, ulist) < 0) {
            fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
            return -1;
        }


        int layers_ref;
        sprintf(varname, "layers_%i", s);
        read_hdf5_attribute(file, varname, H5T_NATIVE_INT, &layers_ref);

        assert(layers_ref == layers);   

        int uperms[layers][nqubits];
        for (int i = 0; i < layers; i++)
        {
            sprintf(varname, "uperm%i_%i", i, s);
            if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
                fprintf(stderr, "reading permutation data from disk failed\n");
                return -1;
            }
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

        const int* upperms[layers];
        for (int l = 0; l < layers; l++) {
            upperms[l] = uperms[l];
        }

        apply_matchgate_brickwall_unitary(ulist, layers, upperms, &psi0, &Upsi);


        double norm_error = uniform_distance(n, Upsi_ref.data, Upsi.data);
        
        printf("layers: %i | %lf\n", layers, norm_error);
        errors[s - 1] = norm_error;
        layers_arr[s - 1] = layers;
    }

    assert(layers_arr[us - 1] == ulayers);

    sprintf(filename, "../examples/spl_hubbard2d/test_splitting/error_out/spl_hubbard2d_%s_q%i_us%i_u%i_t%.2fs_g%.2f_errors.hdf5", model, nqubits, us, ulayers, t, g);
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

    
	hsize_t dims[1] = { us };
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

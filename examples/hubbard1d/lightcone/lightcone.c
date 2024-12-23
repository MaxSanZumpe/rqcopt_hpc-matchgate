#include <assert.h>
#include <complex.h>
#include "statevector.h"
#include "matchgate_brickwall.h"
#include "gate.h"
#include "matchgate.h"
#include "util.h"

#ifdef LRZ_HPC
	#include <mkl_cblas.h>
#else
	#include <cblas.h>
#endif

void apply_oneqgate(const struct mat2x2* gate, const int i, const struct statevector* psi, struct statevector* psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);

    const intqs m = (intqs)1 << i;
    const intqs n = (intqs)1 << (psi->nqubits - 1 - i);

    for (intqs a = 0; a < m; a++)
    {
        for (intqs b = 0; b < n; b++)
        {
            numeric x = psi->data[(a*2    )*n + b];
            numeric y = psi->data[(a*2 + 1)*n + b];

            psi_out->data[(a*2    )*n + b] = gate->data[ 0] * x + gate->data[ 1] * y;
            psi_out->data[(a*2 + 1)*n + b] = gate->data[ 2] * x + gate->data[ 3] * y; 
        }
    }
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


void print_state(struct statevector* vec) {
    const intqs n = (intqs)1 << vec->nqubits;
    for (intqs a = 0; a < n; a++) {
        printf("Re: %lf Im: %lf\n", creal(vec->data[a]), cimag(vec->data[a]));
    }
}

int main()
{   
    const int nlayers = 9;
    const int nqubits = 16;

    const int ulayers = 561;
	const int order = 2;

    float g = 4.0;
    float delta = 0.02;

    const int num_steps = 200;

    const intqs n = (intqs)1 << nqubits;
    struct statevector Spsi, Upsi, Npsi, Ipsi, Jpsi;
    if (allocate_statevector(nqubits, &Spsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Upsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Npsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Ipsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
    if (allocate_statevector(nqubits, &Jpsi) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }

	struct matchgate* ulist = aligned_alloc(MEM_DATA_ALIGN, ulayers * sizeof(struct matchgate));
    
    char filename[1024];
	sprintf(filename, "../examples/hubbard1d/lightcone/hubbard1d_suzuki%i_n%i_q%i_u%i_t%.2fs_g%.2f_init.hdf5", order, nlayers, nqubits, ulayers, delta, g);
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

    if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, Spsi.data) < 0) {
		fprintf(stderr, "reading input statevector data from disk failed");
		return -1;
	}

    numeric* expiH;
    char varname[32];
    const int* upperms[ulayers];
    int uperms[ulayers][nqubits];
    if (ulayers == 0) {
        expiH = aligned_alloc(MEM_DATA_ALIGN, n * n * sizeof(numeric));
	    if (expiH == NULL) {
		    fprintf(stderr, "memory allocation for target unitary failed\n");
		    return -1;
	    }
	    if (read_hdf5_dataset(file, "expiH", H5T_NATIVE_DOUBLE, expiH) < 0) {
		    fprintf(stderr, "reading 'expiH' from disk failed\n");
		    return -1;
        }
    } else {
        if (read_hdf5_dataset(file, "ulist", H5T_NATIVE_DOUBLE, ulist) < 0) {
            fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
            return -1;
        }
	
        for (int i = 0; i < ulayers; i++)
        {
            sprintf(varname, "uperm%i", i);
            if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
                fprintf(stderr, "reading permutation data from disk failed\n");
                return -1;
            }
        }

        for (int l = 0; l < ulayers; l++) {
	        upperms[l] = uperms[l];
        }
    }

    int nqubits_ref;
    if (read_hdf5_attribute(file, "nqubits", H5T_NATIVE_INT, &nqubits_ref) < 0) {
        fprintf(stderr, "reading qubit number from disk failed\n");
		return -1;
    }

    int ulayers_ref;
    if (read_hdf5_attribute(file, "ulayers", H5T_NATIVE_INT, &ulayers_ref) < 0) {
        fprintf(stderr, "reading layer number from disk failed\n");
		return -1;
    }

    if (read_hdf5_attribute(file, "t", H5T_NATIVE_FLOAT, &delta) < 0) {
        fprintf(stderr, "reading time step from disk failed\n");
		return -1;
    }

	H5Fclose(file);

    assert (nqubits == nqubits_ref);
    assert (ulayers == ulayers_ref);

    struct mat2x2 number_operator = { .data = { 0, 0, 0, 1 } };


    numeric f[num_steps + 1][nqubits/2];

    int i = 0;
    apply_oneqgate(&number_operator, i, &Spsi, &Ipsi);
    numeric norm = 0;
    for (intqs a = 0; a < n; a++) {
        norm += conj(Spsi.data[a]) * Ipsi.data[a];
    }

    struct mat2x2 avg_number_operator = { .data = { -norm, 0, 0, 1 - norm } };

    apply_oneqgate(&avg_number_operator, i, &Spsi, &Ipsi);
    
    for (int t = 0; t < num_steps + 1; t++) {

        if (t == 0) {
            for (intqs b = 0; b < n; b++) {
                Upsi.data[b] = Spsi.data[b];
                Npsi.data[b] = Ipsi.data[b];
            }
        } else {

            if (ulayers == 0) {
                ufunc(&Spsi, expiH, &Upsi);
                ufunc(&Ipsi, expiH, &Npsi);
            } else {
                apply_matchgate_brickwall_unitary(ulist, ulayers, upperms, &Spsi, &Upsi);
                apply_matchgate_brickwall_unitary(ulist, ulayers, upperms, &Ipsi, &Npsi);
            }
                

            for (intqs b = 0; b < n; b++) {
                Spsi.data[b] = Upsi.data[b];
                Ipsi.data[b] = Npsi.data[b];
            }
        }

        for (int j = 0; j < nqubits/2; j++) {
            apply_oneqgate(&avg_number_operator, j, &Npsi, &Jpsi);

            f[t][j] = 0;
            for (intqs a = 0; a < n; a++) {
                f[t][j] += conj(Upsi.data[a]) * Jpsi.data[a];
            } 

            printf("t = %i, j = %i : ", t, j);
            printf("Re: %lf, Im: %lf\n", creal(f[t][j]), cimag(f[t][j]));
        }
    }

    sprintf(filename, "../examples/hubbard1d/lightcone/hubbard1d_q%i_g%.2f_u%i_lightcone.hdf5", nqubits, g, ulayers);
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

    for (int t = 0; t < num_steps + 1; t++) {
		sprintf(varname, "f_tstep%i", t);
        hsize_t dims[2] = { nqubits/2, 2 };
        if (write_hdf5_dataset(file, varname, 2, dims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, f[t]) < 0) {
            fprintf(stderr, "writing obs to disk failed\n");
            return -1;
        }
    }

    if (write_hdf5_scalar_attribute(file, "nqubits", H5T_STD_I32LE, H5T_NATIVE_INT, &nqubits)) {
		fprintf(stderr, "writing 'nqubits' to disk failed\n");
		return -1;
	}

    if (write_hdf5_scalar_attribute(file, "delta", H5T_IEEE_F64LE, H5T_NATIVE_FLOAT, &delta) < 0) {
		fprintf(stderr, "writing time step to disk failed\n");
		return -1;
	}

    H5Fclose(file);

    aligned_free(ulist);
    free_statevector(&Spsi);
    free_statevector(&Upsi);
    free_statevector(&Npsi);
    free_statevector(&Ipsi);
    free_statevector(&Jpsi);
}

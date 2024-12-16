#include <memory.h>
#include <assert.h>
#include "matchgate_target.h"
#include "matchgate_brickwall.h"
#include "numerical_gradient.h"
#include "util.h"

#ifdef LRZ_HPC
	#include <mkl_cblas.h>
#else
	#include <cblas.h>
#endif

#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


static int ufunc(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);

	const intqs n = (intqs)1 << psi->nqubits;
	for (intqs i = 0; i < n; i++)
	{
		#ifndef COMPLEX_CIRCUIT
		psi_out->data[i] = -1.1 * psi->data[((i + 3) * 113) % n] - 0.7 * psi->data[((i + 9) * 173) % n] + 0.5 * psi->data[i] + 0.3 * psi->data[((i + 4) * 199) % n];
		#else
		psi_out->data[i] = (-1.1 + 0.8*I) * psi->data[((i + 3) * 113) % n] + (0.4 - 0.7*I) * psi->data[((i + 9) * 173) % n] + (0.5 + 0.1*I) * psi->data[i] + (-0.3 + 0.2*I) * psi->data[((i + 4) * 199) % n];
		#endif
	}

	return 0;
}

static int ufunc2(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out)
{
	const intqs n = (intqs)1 << psi->nqubits;
	const numeric* U = (numeric*)fdata;

	// apply U
	numeric alpha = 1;
	numeric beta  = 0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &alpha, U, n, psi->data, 1, &beta, psi_out->data, 1);

	return 0;
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

char* test_matchgate_circuit_unitary_target()
{
	const int nqubits = 7;
	const int ngates  = 5;

	hid_t file = H5Fopen("../test/data/test_matchgate_circuit_unitary_target" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_matchgate_circuit_unitary_target failed";
	}

	struct matchgate* gates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct matchgate));
	for (int i = 0; i < ngates; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "G/center%i", i);
        sprintf(varname2, "G/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, gates[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
        if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, gates[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	if (read_hdf5_dataset(file, "wires", H5T_NATIVE_INT, wires) < 0) {
		return "reading wire indices from disk failed";
	}

	numeric f;
	if (matchgate_circuit_unitary_target(ufunc, NULL, gates, ngates, wires, nqubits, &f) < 0) {
		return "'circuit_unitary_target' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	aligned_free(wires);
	aligned_free(gates);

	H5Fclose(file);

	return 0;
}


char* test_matchgate_brickwall_unitary_target()
{
	const int nqubits = 8;

	hid_t file = H5Fopen("../test/data/test_matchgate_brickwall_unitary_target" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_matchgate_brickwall_unitary_target failed";
	}

	struct matchgate vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname1[32],  varname2[32];
		sprintf(varname1, "V/center%i", i);
        sprintf(varname2, "V/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, vlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
        if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, vlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		numeric f;
		if (matchgate_brickwall_unitary_target(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f) < 0) {
			return "'brickwall_unitary_target' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		numeric f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (_abs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}
	}

	H5Fclose(file);

	return 0;
}


struct brickwall_unitary_target_params
{
	linear_func ufunc;
	const int** perms;
	int nqubits;
	int nlayers;
};

// wrapper of brickwall unitary target function
static void matchgate_brickwall_unitary_target_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_target_params* params = p;

	struct matchgate* vlist = aligned_alloc(MEM_DATA_ALIGN, params->nlayers * sizeof(struct matchgate));
	for (int i = 0; i < params->nlayers; i++) {
		memcpy(vlist[i].center_block.data, &x[i * 8    ], sizeof(vlist[i].center_block.data));
		memcpy(vlist[i].corner_block.data, &x[i * 8 + 4], sizeof(vlist[i].corner_block.data));
	}

	numeric f;
	matchgate_brickwall_unitary_target(params->ufunc, NULL, vlist, params->nlayers, params->nqubits, params->perms, &f);
	*y = f;

	aligned_free(vlist);
}

// wrapper of brickwall unitary target gradient function
static void matchgate_brickwall_unitary_target_gradient_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_target_params* params = p;

	numeric f;
	matchgate_brickwall_unitary_target_and_gradient(params->ufunc, NULL, (struct matchgate*)x, params->nlayers, params->nqubits, params->perms, &f, (struct matchgate*)y);
}

char* test_matchgate_brickwall_unitary_target_gradient_hessian()
{
	const int nqubits = 6;

	hid_t file = H5Fopen("../test/data/test_matchgate_brickwall_unitary_target_gradient_hessian" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_gradient_hessian failed";
	}

	struct matchgate vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "V/center%i", i);
		sprintf(varname2, "V/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, vlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, vlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	// gradient direction of quantum gates
	struct matchgate Zlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "Z/center%i", i);
		sprintf(varname2, "Z/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, Zlist[i].center_block.data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, Zlist[i].corner_block.data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
	}

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{	
		const int m = nlayers[i] * 8;

		numeric f;
		struct matchgate dvlist[5];
		numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		
		if (matchgate_brickwall_unitary_target_gradient_hessian(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f, dvlist, hess) < 0) {
			return "'brickwall_unitary_target_gradient_hessian' failed internally";
		}
		
		// check symmetry of Hessian matrix
		double err_symm = 0;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				err_symm = fmax(err_symm, _abs(hess[j*m + k] - hess[k*m + j]));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		numeric f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}
		// compare with reference
		if (_abs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}

		const double h = 1e-5;
		struct brickwall_unitary_target_params params = {
			.ufunc   = ufunc,
			.nqubits = nqubits,
			.nlayers = nlayers[i],
			.perms   = pperms,
		};

		// numerical gradient
		struct matchgate dvlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(matchgate_brickwall_unitary_target_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		// #else
		// // numerical_gradient_backward(matchgate_brickwall_unitary_target_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(8, (numeric*)&dvlist[j], (numeric*)&dvlist_num[j]) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}

		}

		int o;
		char varname1[32], varname2[32];
		struct matchgate dvlist_ref[5];
		for (o = 0; o < nlayers[i]; o++){
			sprintf(varname1, "dVlist%i/center%i", i, o);
			sprintf(varname2, "dVlist%i/corner%i", i, o);
			if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, dvlist_ref[o].center_block.data) < 0) {
				return "reading reference gradient data from disk failed";
			}
			if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, dvlist_ref[o].corner_block.data) < 0) {
				return "reading reference gradient data from disk failed";
			}
		}	

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(8, (numeric*)&dvlist[j], (numeric*)&dvlist_ref[j]) > 1e-12) {
				return "computed brickwall circuit gradient does not match reference";
			}
		}

		// numerical gradient
		struct matchgate dVZlist_num[5];
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(matchgate_brickwall_unitary_target_gradient_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, nlayers[i] * 8, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#else
		numerical_gradient_backward(matchgate_brickwall_unitary_target_gradient_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, nlayers[i] * 8, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#endif
		// Hessian matrix times gradient direction
		struct matchgate dVZlist[5];
		#ifdef COMPLEX_CIRCUIT
		numeric alpha = 1;
		numeric beta  = 0;
		cblas_zgemv(CblasRowMajor, CblasNoTrans, m, m, &alpha, hess, m, (numeric*)Zlist, 1, &beta, (numeric*)dVZlist, 1);
		#else
		cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, hess, m, (numeric*)Zlist, 1, 0, (numeric*)dVZlist, 1);
		#endif
		// compare
		if (uniform_distance(m, (numeric*)dVZlist, (numeric*)dVZlist_num) > 1e-7) {
			return "second derivative with respect to gates computed by 'unitary_target_gradient_hessian' does not match finite difference approximation";
		}

		// sprintf(varname, "hess%i", i);
		// numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		// if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, hess_ref) < 0) {
		// 	return "reading reference Hessian matrix from disk failed";
		// }

		// // compare with reference
		// if (uniform_distance(m * m, hess, hess_ref) > 1e-12) {
		// 	return "computed brickwall circuit Hessian matrix does not match reference";
		// }

		//aligned_free(hess_ref);
		aligned_free(hess);
	}

	H5Fclose(file);

	return 0;
}


char* test_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix()
{
	const int nqubits = 6;
	const int nlayers = 5;

	hid_t file = H5Fopen("../test/data/test_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix failed";
	}

	struct matchgate vlist[5];
	for (int i = 0; i < nlayers; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "V/center%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, vlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}

		sprintf(varname2, "V/corner%i", i);
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, vlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	const int m = nlayers * 8;

	numeric f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	double* H = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));

	if (matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix(ufunc, NULL, vlist, nlayers, nqubits, pperms, &f, grad, H) < 0) {
		return "matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	double* grad_ref = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	if (uniform_distance_real(m, grad, grad_ref) > 1e-14) {
		return "computed gate gradient vector does not match reference";
	}

	// check symmetry
	double es = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			es = fmax(es, fabs(H[i * m + j] - H[j * m + i]));
		}
	}
	if (es > 1e-14) {
		return "computed gate Hessian matrix is not symmetric";
	}

	double* H_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));
	if (H_ref == NULL) {
		return "memory allocation for reference Hessian matrix failed";
	}
	if (read_hdf5_dataset(file, "H", H5T_NATIVE_DOUBLE, H_ref) < 0) {
		return "reading reference Hessian matrix from disk failed";
	}

	double es2 = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			es = fmax(es2, fabs(H_ref[i * m + j] - H_ref[j * m + i]));
		}
	}
	if (es > 1e-14) {
		return "reference gate Hessian matrix is not symmetric";
	}

	// compare with reference
	if (uniform_distance_real(m * m, H, H_ref) > 1e-13) {
		return "computed unitary target Hessian matrix does not match reference";
	}

	aligned_free(H_ref);
	aligned_free(grad_ref);
	aligned_free(H);
	aligned_free(grad);

	H5Fclose(file);

	return 0;
}


//-----------------------------------------------------------------------------------------------------
// Testing computing hessian with translation invariance in the target unitary
//
//

char* test_matchgate_ti_brickwall_unitary_target_gradient_hessian()
{
	const int nqubits = 8;
	const int nlayers = 7;
	const intqs n = (intqs)1 << nqubits;

	hid_t file = H5Fopen("../test/data/test_matchgate_ti_brickwall_unitary_target_gradient_hessian" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_gradient_hessian failed";
	}
		
	// translational invariant target unitary
	numeric* expiH = aligned_alloc(MEM_DATA_ALIGN, n * n * sizeof(numeric));
	if (expiH == NULL) {
		fprintf(stderr, "memory allocation for target unitary failed\n");
	}
	if (read_hdf5_dataset(file, "expiH", H5T_NATIVE_DOUBLE, expiH) < 0) {
		fprintf(stderr, "reading 'expiH' from disk failed\n");
	}

	struct matchgate vlist[nlayers];
	for (int i = 0; i < nlayers; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "V/center%i", i);
		sprintf(varname2, "V/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, vlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, vlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[nlayers][nqubits];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}

	const int* pperms[nlayers];
	for (int i = 0; i < nlayers; i++) {
		pperms[i] = perms[i];
	}

	const int m = nlayers * nqubits;

	numeric f_ref;
	struct matchgate dvlist_ref[nlayers];
	numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));

	if (matchgate_brickwall_unitary_target_gradient_hessian(ufunc2, expiH, vlist, nlayers, nqubits, pperms, &f_ref, dvlist_ref, hess_ref) < 0) {
		return "'matchgate_brickwall_unitary_target_gradient_hessian' failed internally";
	}

	numeric f;
	struct matchgate dvlist[nlayers];
	numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	
	
	if (matchgate_ti_brickwall_unitary_target_gradient_hessian(ufunc2, expiH, vlist, nlayers, nqubits, pperms, &f, dvlist, hess) < 0) {
		return "'matchgate_ti_brickwall_unitary_target_gradient_hessian' failed internally";
	}

	// compare gradients
	for (int j = 0; j < nlayers; j++) {
		if (uniform_distance(8, (numeric*)&dvlist[j], (numeric*)&dvlist_ref[j]) > 1e-8) {
			return "target function gradient with respect to gates does not match finite difference approximation";
		}
	}

	// compare hessians
	if (uniform_distance(m * m, hess, hess_ref) > 1e-13) {
		return "computed unitary target Hessian matrix does not match reference";
	}

	aligned_free(hess_ref);
	aligned_free(hess);
	
	H5Fclose(file);

	return 0;
}


char* test_matchgate_ti_brickwall_unitary_target_gradient_hessian2()
{
	const int nqubits = 8;
	const int nlayers = 5;
	const intqs n = (intqs)1 << nqubits;

	hid_t file = H5Fopen("../test/data/test_matchgate_ti_brickwall_unitary_target_gradient_hessian2" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_gradient_hessian failed";
	}
		
	// translational invariant target unitary
	numeric* expiH = aligned_alloc(MEM_DATA_ALIGN, n * n * sizeof(numeric));
	if (expiH == NULL) {
		fprintf(stderr, "memory allocation for target unitary failed\n");
	}
	if (read_hdf5_dataset(file, "expiH", H5T_NATIVE_DOUBLE, expiH) < 0) {
		fprintf(stderr, "reading 'expiH' from disk failed\n");
	}

	struct matchgate vlist[nlayers];
	if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, vlist) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
	}

	int perms[nlayers][nqubits];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}

	const int* pperms[nlayers];
	for (int i = 0; i < nlayers; i++) {
		pperms[i] = perms[i];
	}

	const int m = nlayers * nqubits;

	numeric f_ref;
	struct matchgate dvlist_ref[nlayers];
	numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));

	if (matchgate_brickwall_unitary_target_gradient_hessian(ufunc2, expiH, vlist, nlayers, nqubits, pperms, &f_ref, dvlist_ref, hess_ref) < 0) {
		return "'matchgate_brickwall_unitary_target_gradient_hessian' failed internally";
	}

	numeric f;
	struct matchgate dvlist[nlayers];
	numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	
	
	if (matchgate_ti_brickwall_unitary_target_gradient_hessian(ufunc2, expiH, vlist, nlayers, nqubits, pperms, &f, dvlist, hess) < 0) {
		return "'matchgate_ti_brickwall_unitary_target_gradient_hessian' failed internally";
	}

	// compare gradients
	for (int j = 0; j < nlayers; j++) {
		if (uniform_distance(8, (numeric*)&dvlist[j], (numeric*)&dvlist_ref[j]) > 1e-8) {
			return "target function gradient with respect to gates does not match finite difference approximation";
		}
	}

	// compare hessians
	if (uniform_distance(m * m, hess, hess_ref) > 1e-12) {
		return "computed unitary target Hessian matrix does not match reference";
	}

	aligned_free(hess_ref);
	aligned_free(hess);
	
	H5Fclose(file);

	return 0;
}


char* test_matchgate_ti_brickwall_unitary_target_gradient_hessian3()
{
	const int nqubits = 8;
	const int nlayers = 7;
	const int ulayers = 201;
	const intqs n = (intqs)1 << nqubits;

	
	hid_t file = H5Fopen("../test/data/test_matchgate_ti_brickwall_unitary_target_gradient_hessian3" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_gradient_hessian failed";
	}
	
	struct matchgate vlist[nlayers];
	if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, vlist) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	struct matchgate ulist[ulayers];
	if (read_hdf5_dataset(file, "ulist", H5T_NATIVE_DOUBLE, ulist) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	char varname[32];
	int perms[nlayers][nqubits];
	for (int i = 0; i < nlayers; i++) {
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}

	int uperms[ulayers][nqubits];
	for (int i = 0; i < ulayers; i++) {
		sprintf(varname, "uperm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, uperms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}

	H5Fclose(file);

	const int* upperms[ulayers];
	for (int i = 0; i < ulayers; i++) {
		upperms[i] = uperms[i];
	}

	const int* pperms[nlayers];
	for (int i = 0; i < nlayers; i++) {
		pperms[i] = perms[i];
	}


	struct u_splitting udata = {
		.ulist   = ulist,
		.ulayers = ulayers,
		.upperms = upperms,
	};


	const int m = nlayers * 8;

	
	numeric f, f_ref;
	struct matchgate dvlist[nlayers];
	struct matchgate dvlist_ref[nlayers];

	numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));

	
	if (matchgate_brickwall_unitary_target_gradient_hessian(ufunc_matfree, &udata, vlist, nlayers, nqubits, pperms, &f_ref, dvlist_ref, hess_ref) < 0) {
		return "'mat4x4_brickwall_unitary_target_gradient_hessian' failed internally";
	}

	if (matchgate_ti_brickwall_unitary_target_gradient_hessian(ufunc_matfree, &udata, vlist, nlayers, nqubits, pperms, &f, dvlist, hess) < 0) {
		return "'mat4x4_ti_brickwall_unitary_target_gradient_hessian' failed internally";
	}

	// compare gradients
	for (int j = 0; j < nlayers; j++) {
		if (uniform_distance(8, (numeric*)&dvlist[j], (numeric*)&dvlist_ref[j]) > 1e-8) {
			return "translational invariance (usplit): target function gradient with respect to gates does not match finite difference approximation";
		}
	}

	// compare hessians
	if (uniform_distance(m * m, hess, hess_ref) > 1e-12) {
		return "tranlation invariance (usplit): computed unitary target Hessian matrix does not match reference";
	}


	aligned_free(hess_ref);
	aligned_free(hess);


	return 0;
}



char* test_parallel_matchgate_brickwall_unitary_target_gradient_hessian()
{
	const int nqubits = 6;

	hid_t file = H5Fopen("../test/data/test_matchgate_brickwall_unitary_target_gradient_hessian" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_gradient_hessian failed";
	}

	struct matchgate vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "V/center%i", i);
		sprintf(varname2, "V/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, vlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, vlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	// gradient direction of quantum gates
	struct matchgate Zlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "Z/center%i", i);
		sprintf(varname2, "Z/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, Zlist[i].center_block.data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, Zlist[i].corner_block.data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
	}

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{	
		const int m = nlayers[i] * 8;

		numeric f;
		struct matchgate dvlist[5];
		numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		
		if (parallel_matchgate_brickwall_unitary_target_gradient_hessian(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f, dvlist, hess) < 0) {
			return "'brickwall_unitary_target_gradient_hessian' failed internally";
		}
		
		// check symmetry of Hessian matrix
		double err_symm = 0;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				err_symm = fmax(err_symm, _abs(hess[j*m + k] - hess[k*m + j]));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		numeric f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}
		// compare with reference
		if (_abs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}

		const double h = 1e-5;
		struct brickwall_unitary_target_params params = {
			.ufunc   = ufunc,
			.nqubits = nqubits,
			.nlayers = nlayers[i],
			.perms   = pperms,
		};

		// numerical gradient
		struct matchgate dvlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(matchgate_brickwall_unitary_target_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		// #else
		// // numerical_gradient_backward(matchgate_brickwall_unitary_target_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(8, (numeric*)&dvlist[j], (numeric*)&dvlist_num[j]) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}

		}

		int o;
		char varname1[32], varname2[32];
		struct matchgate dvlist_ref[5];
		for (o = 0; o < nlayers[i]; o++){
			sprintf(varname1, "dVlist%i/center%i", i, o);
			sprintf(varname2, "dVlist%i/corner%i", i, o);
			if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, dvlist_ref[o].center_block.data) < 0) {
				return "reading reference gradient data from disk failed";
			}
			if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, dvlist_ref[o].corner_block.data) < 0) {
				return "reading reference gradient data from disk failed";
			}
		}	

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(8, (numeric*)&dvlist[j], (numeric*)&dvlist_ref[j]) > 1e-12) {
				return "computed brickwall circuit gradient does not match reference";
			}
		}

		// numerical gradient
		struct matchgate dVZlist_num[5];
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(matchgate_brickwall_unitary_target_gradient_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, nlayers[i] * 8, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#else
		numerical_gradient_backward(matchgate_brickwall_unitary_target_gradient_wrapper, &params, nlayers[i] * 8, (numeric*)vlist, nlayers[i] * 8, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#endif
		// Hessian matrix times gradient direction
		struct matchgate dVZlist[5];
		#ifdef COMPLEX_CIRCUIT
		numeric alpha = 1;
		numeric beta  = 0;
		cblas_zgemv(CblasRowMajor, CblasNoTrans, m, m, &alpha, hess, m, (numeric*)Zlist, 1, &beta, (numeric*)dVZlist, 1);
		#else
		cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, hess, m, (numeric*)Zlist, 1, 0, (numeric*)dVZlist, 1);
		#endif
		// compare
		if (uniform_distance(m, (numeric*)dVZlist, (numeric*)dVZlist_num) > 1e-7) {
			return "second derivative with respect to gates computed by 'unitary_target_gradient_hessian' does not match finite difference approximation";
		}

		// sprintf(varname, "hess%i", i);
		// numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		// if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, hess_ref) < 0) {
		// 	return "reading reference Hessian matrix from disk failed";
		// }

		// // compare with reference
		// if (uniform_distance(m * m, hess, hess_ref) > 1e-12) {
		// 	return "computed brickwall circuit Hessian matrix does not match reference";
		// }

		//aligned_free(hess_ref);
		aligned_free(hess);
	}

	H5Fclose(file);

	return 0;
}
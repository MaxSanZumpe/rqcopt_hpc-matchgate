#include <memory.h>
#include <assert.h>
#include "config.h"
#include "matchgate.h"
#include "statevector.h"
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

char* test_apply_matchgate_brickwall_unitary()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_apply_matchgate_brickwall_unitary" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_matchgate_brickwall_unitary failed";
	}

	struct statevector psi, chi, chi_ref;
	if (allocate_statevector(L, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi_ref) < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct matchgate Mlist[4];
	for (int i = 0; i < 4; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "v/center%i", i);
		sprintf(varname2, "v/corner%i", i);

		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, Mlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, Mlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		// apply the brickwall unitary
		if (apply_matchgate_brickwall_unitary(Mlist, nlayers, pperms, &psi, &chi) < 0) {
			return "'apply_matchgae_brickwall_unitary' failed internally";
		}

		char varname[32];
		sprintf(varname, "chi%i", nlayers - 3);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, chi_ref.data) < 0) {
			return "reading output statevector data from disk failed";
		}

		// compare with reference
		if (uniform_distance((long)1 << L, chi.data, chi_ref.data) > 1e-12) {
			return "quantum state after applying gate does not match reference";
		}
	}

	free_statevector(&chi_ref);
	free_statevector(&chi);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


struct matchgate_brickwall_unitary_forward_psi_params
{
	int nqubits;
	int nlayers;
	const struct matchgate* Mlist;
	const int** perms;
};

// wrapper of brickwall_unitary_forward as a function of 'psi'
static void matchgate_brickwall_unitary_forward_psi(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct matchgate_brickwall_unitary_forward_psi_params* params = p;

	struct statevector psi;
	allocate_statevector(params->nqubits, &psi);
	memcpy(psi.data, x, ((size_t)1 << params->nqubits) * sizeof(numeric));

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	struct quantum_circuit_cache cache;
	allocate_quantum_circuit_cache(params->nqubits, params->nlayers * (params->nqubits / 2), &cache);

	matchgate_brickwall_unitary_forward(params->Mlist, params->nlayers, params->perms, &psi, &cache, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_quantum_circuit_cache(&cache);
	free_statevector(&psi_out);
	free_statevector(&psi);
}

struct matchgate_brickwall_unitary_forward_gates_params
{
	int nqubits;
	int nlayers;
	const struct statevector* psi;
	const int** perms;
};

// wrapper of brickwall_unitary_forward as a function of the gates
static void matchgate_brickwall_unitary_forward_gates(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct matchgate_brickwall_unitary_forward_gates_params* params = p;

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	struct quantum_circuit_cache cache;
	allocate_quantum_circuit_cache(params->nqubits, params->nlayers * (params->nqubits / 2), &cache);

	matchgate_brickwall_unitary_forward((struct matchgate*)x, params->nlayers, params->perms, params->psi, &cache, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_quantum_circuit_cache(&cache);
	free_statevector(&psi_out);
}

char* test_matchgate_brickwall_unitary_backward()
{
	int L = 6;

	hid_t file = H5Fopen("../test/data/test_matchgate_brickwall_unitary_backward" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_matchgate_brickwall_unitary_backward failed";
	}

	struct statevector psi, psi_out, psi_out_ref, dpsi_out, dpsi;
	if (allocate_statevector(L, &psi)         < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &psi_out)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &psi_out_ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &dpsi_out)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &dpsi)        < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	if (read_hdf5_dataset(file, "dpsi_out", H5T_NATIVE_DOUBLE, dpsi_out.data) < 0) {
		return "reading upstream gradient data from disk failed";
	}

	struct matchgate Mlist[4];
	for (int i = 0; i < 4; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "v/center%i", i);
		sprintf(varname2, "v/corner%i", i);

		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, Mlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, Mlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		struct quantum_circuit_cache cache;
		if (allocate_quantum_circuit_cache(L, nlayers * (L / 2), &cache) < 0) {
			return "'allocate_quantum_circuit_cache' failed";
		}

		// brickwall unitary forward pass
		if (matchgate_brickwall_unitary_forward(Mlist, nlayers, pperms, &psi, &cache, &psi_out) < 0) {
			return "'matchgate_brickwall_unitary_forward' failed internally";
		}

		char varname[32];
		sprintf(varname, "psi_out%i", nlayers - 3);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
			return "reading output statevector data from disk failed";
		}
		// compare output state of forward pass with reference
		if (uniform_distance((long)1 << L, psi_out.data, psi_out_ref.data) > 1e-12) {
			return "quantum state after applying brick wall quantum circuit does not match reference";
		}

		// brickwall unitary backward pass
		struct matchgate dVlist[4];
		if (matchgate_brickwall_unitary_backward(Mlist, nlayers, pperms, &cache, &dpsi_out, &dpsi, dVlist) < 0) {
			return "'matchgate_brickwall_unitary_backward' failed internally";
		}

		const double h = 1e-5;

		// numerical gradient with respect to 'psi'
		struct matchgate_brickwall_unitary_forward_psi_params params_psi = {
			.nqubits = L,
			.nlayers = nlayers,
			.Mlist = Mlist,
			.perms = pperms,
		};
		struct statevector dpsi_num;
		if (allocate_statevector(L, &dpsi_num) < 0) { return "memory allocation failed"; }
		numerical_gradient_backward(matchgate_brickwall_unitary_forward_psi, &params_psi, 1 << L, psi.data, 1 << L, dpsi_out.data, h, dpsi_num.data);
		// compare
		if (uniform_distance((long)1 << L, dpsi.data, dpsi_num.data) > 1e-8) {
			return "gradient with respect to 'psi' computed by 'brickwall_unitary_backward' does not match finite difference approximation";
		}

		// numerical gradient with respect to gates
		struct matchgate_brickwall_unitary_forward_gates_params params_gates = {
			.nqubits = L,
			.nlayers = nlayers,
			.psi = &psi,
			.perms = pperms,
		};
		struct matchgate dVlist_num[4];
		numerical_gradient_backward(matchgate_brickwall_unitary_forward_gates, &params_gates, nlayers * 8, (numeric*)Mlist, 1 << L, dpsi_out.data, h, (numeric*)dVlist_num);
		// compare
		int it;
		for (it = 0; it < nlayers; it++){
			if (uniform_distance(4, dVlist[it].center_block.data, dVlist_num[it].center_block.data) > 1e-8 ||
				uniform_distance(4, dVlist[it].corner_block.data, dVlist_num[it].corner_block.data) > 1e-8) {
				return "gradient with respect to gates computed by 'matchgate_brickwall_unitary_backward' does not match finite difference approximation";
			}
		}

		free_statevector(&dpsi_num);
		free_quantum_circuit_cache(&cache);
	}

	free_statevector(&dpsi);
	free_statevector(&dpsi_out);
	free_statevector(&psi_out_ref);
	free_statevector(&psi_out);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


struct matchgate_brickwall_unitary_derivative_gates_params
{
	int nqubits;
	int nlayers;
	const struct statevector* psi;
	const struct statevector* dpsi_out;
	const int** perms;
};

static void matchgate_brickwall_unitary_derivative_gates(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct matchgate_brickwall_unitary_derivative_gates_params* params = p;

	struct quantum_circuit_cache cache;
	allocate_quantum_circuit_cache(params->nqubits, params->nlayers * (params->nqubits / 2), &cache);

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	// brickwall unitary forward pass
	matchgate_brickwall_unitary_forward((struct matchgate*)x, params->nlayers, params->perms, params->psi, &cache, &psi_out);

	struct statevector dpsi;
	allocate_statevector(params->nqubits, &dpsi);

	// brickwall unitary backward pass
	matchgate_brickwall_unitary_backward((struct matchgate*)x, params->nlayers, params->perms, &cache, params->dpsi_out, &dpsi, (struct matchgate*)y);

	free_statevector(&dpsi);
	free_statevector(&psi_out);
	free_quantum_circuit_cache(&cache);
}

char* test_matchgate_brickwall_unitary_backward_hessian()
{
	int L = 6;

	hid_t file = H5Fopen("../test/data/test_matchgate_brickwall_unitary_backward_hessian" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_matchgate_brickwall_unitary_backward_hessian failed";
	}

	struct statevector psi, psi_out, psi_out_ref, dpsi_out, dpsi;
	if (allocate_statevector(L, &psi)         < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &psi_out)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &psi_out_ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &dpsi_out)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &dpsi)        < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	if (read_hdf5_dataset(file, "dpsi_out", H5T_NATIVE_DOUBLE, dpsi_out.data) < 0) {
		return "reading upstream gradient data from disk failed";
	}

	struct matchgate Mlist[4];
	for (int i = 0; i < 4; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "v/center%i", i);
		sprintf(varname2, "v/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, Mlist[i].center_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, Mlist[i].corner_block.data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	// gradient direction of quantum gates
	struct matchgate Zlist[4];
	for (int i = 0; i < 4; i++)
	{
		char varname1[32], varname2[32];
		sprintf(varname1, "z/center%i", i);
		sprintf(varname2, "z/corner%i", i);
		if (read_hdf5_dataset(file, varname1, H5T_NATIVE_DOUBLE, Zlist[i].center_block.data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
		if (read_hdf5_dataset(file, varname2, H5T_NATIVE_DOUBLE, Zlist[i].corner_block.data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
	}

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		const int m = nlayers * 8;

		struct quantum_circuit_cache cache;
		if (allocate_quantum_circuit_cache(L, nlayers * (L / 2), &cache) < 0) {
			return "'allocate_quantum_circuit_cache' failed";
		}

		// brickwall unitary forward pass
		if (matchgate_brickwall_unitary_forward(Mlist, nlayers, pperms, &psi, &cache, &psi_out) < 0) {
			return "'matchgate_brickwall_unitary_forward' failed internally";
		}

		char varname[32];
		sprintf(varname, "psi_out%i", nlayers - 1);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
			return "reading output statevector data from disk failed";
		}
		// compare output state of forward pass with reference
		if (uniform_distance((long)1 << L, psi_out.data, psi_out_ref.data) > 1e-12) {
			return "quantum state after applying brick wall quantum circuit does not match reference";
		}

		// brickwall unitary backward pass and Hessian computation
		struct matchgate dMlist[4];
		numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (matchgate_brickwall_unitary_backward_hessian(Mlist, nlayers, pperms, &cache, &dpsi_out, &dpsi, dMlist, hess) < 0) {
			return "'brickwall_unitary_backward_hessian' failed internally";
		}

		// check symmetry of Hessian matrix
		double err_symm = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				err_symm = fmax(err_symm, _abs(hess[i*m + j] - hess[j*m + i]));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}

		const double h = 1e-5;

		// numerical gradient with respect to 'psi'
		struct matchgate_brickwall_unitary_forward_psi_params params_psi = {
			.nqubits = L,
			.nlayers = nlayers,
			.Mlist = Mlist,
			.perms = pperms,
		};
		struct statevector dpsi_num;
		if (allocate_statevector(L, &dpsi_num) < 0) { return "memory allocation failed"; }
		numerical_gradient_backward(matchgate_brickwall_unitary_forward_psi, &params_psi, 1 << L, psi.data, 1 << L, dpsi_out.data, h, dpsi_num.data);
		// compare
		if (uniform_distance((long)1 << L, dpsi.data, dpsi_num.data) > 1e-8) {
			return "gradient with respect to 'psi' computed by 'brickwall_unitary_backward_hessian' does not match finite difference approximation";
		}

		// numerical gradient with respect to gates
		struct matchgate_brickwall_unitary_forward_gates_params params_gates = {
			.nqubits = L,
			.nlayers = nlayers,
			.psi = &psi,
			.perms = pperms,
		};
		struct matchgate dMlist_num[4];
		numerical_gradient_backward(matchgate_brickwall_unitary_forward_gates, &params_gates, m, (numeric*)Mlist, 1 << L, dpsi_out.data, h, (numeric*)dMlist_num);
		// compare
		int it;
		for (it = 0; it < nlayers; it++){
			if (uniform_distance(4, dMlist[it].center_block.data, dMlist_num[it].center_block.data) > 1e-8 ||
				uniform_distance(4, dMlist[it].corner_block.data, dMlist_num[it].corner_block.data) > 1e-8) {
				return "gradient with respect to gates computed by 'matchgate_brickwall_unitary_backward' does not match finite difference approximation";
			}
		}

		// numerical second derivative with respect to gates in direction 'Z'
		struct matchgate_brickwall_unitary_derivative_gates_params deriv_gates_params = {
			.nqubits = L,
			.nlayers = nlayers,
			.psi = &psi,
			.dpsi_out = &dpsi_out,
			.perms = pperms,
		};
		struct matchgate dVZlist_num[4];
		numerical_gradient_backward(matchgate_brickwall_unitary_derivative_gates, &deriv_gates_params, m, (numeric*)Mlist, m, (numeric*)Zlist, h, (numeric*)dVZlist_num);
	
		// Hessian matrix times gradient direction
		struct matchgate dVZlist[4];
		numeric alpha = 1;
		numeric beta  = 0;
		cblas_zgemv(CblasRowMajor, CblasNoTrans, m, m, &alpha, hess, m, (numeric*)Zlist, 1, &beta, (numeric*)dVZlist, 1);

		if (uniform_distance(8*nlayers, (numeric*)dVZlist, (numeric*)dVZlist_num) > 1e-8){
			return "second derivative with respect to gates computed by 'matchgate_brickwall_unitary_backward_hessian' does not match finite difference approximation";
		}
		
		
		free_statevector(&dpsi_num);
		aligned_free(hess);
		free_quantum_circuit_cache(&cache);
	}

	free_statevector(&dpsi);
	free_statevector(&dpsi_out);
	free_statevector(&psi_out_ref);
	free_statevector(&psi_out);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


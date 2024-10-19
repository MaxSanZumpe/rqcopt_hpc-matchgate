#include <memory.h>
#include "numerical_gradient.h"
#include "gate.h"
#include "util.h"
#include "matchgate.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif

char* test_apply_matchgate()
{
	int L = 9;

	hid_t file = H5Fopen("../test/data/test_apply_matchgate" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate failed";
	}

	struct statevector psi, chi1, chi1ref, chi2, chi2ref, chi3, chi3ref;
	if (allocate_statevector(L, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3ref) < 0) { return "memory allocation failed"; }

	struct matchgate V;
	if (read_hdf5_dataset(file, "v/center", H5T_NATIVE_DOUBLE, V.center_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	if (read_hdf5_dataset(file, "v/corner", H5T_NATIVE_DOUBLE, V.corner_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi1", H5T_NATIVE_DOUBLE, chi1ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi2", H5T_NATIVE_DOUBLE, chi2ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi3", H5T_NATIVE_DOUBLE, chi3ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}

	// apply the gate
	apply_matchgate(&V, 2, 5, &psi, &chi1);
	apply_matchgate(&V, 4, 1, &psi, &chi2);
	apply_matchgate(&V, 3, 4, &psi, &chi3);

	// compare with reference
	if (uniform_distance((long)1 << L, chi1.data, chi1ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }
	if (uniform_distance((long)1 << L, chi2.data, chi2ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }
	if (uniform_distance((long)1 << L, chi3.data, chi3ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }

	free_statevector(&chi3ref);
	free_statevector(&chi3);
	free_statevector(&chi2ref);
	free_statevector(&chi2);
	free_statevector(&chi1ref);
	free_statevector(&chi1);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}	


struct apply_matchgate_psi_params
{
	int nqubits;
	const struct matchgate* gate;
	int i, j;
};

// wrapper of apply_matchgate as a function of 'psi'
static void apply_matchgate_psi(const numeric* x, void* p, numeric* y)
{
	const struct apply_matchgate_psi_params* params = p;

	struct statevector psi;
	allocate_statevector(params->nqubits, &psi);
	memcpy(psi.data, x, ((size_t)1 << params->nqubits) * sizeof(numeric));

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	apply_matchgate(params->gate, params->i, params->j, &psi, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_statevector(&psi_out);
	free_statevector(&psi);
}

struct apply_matchgate_v_params
{
	const struct statevector* psi;
	int i, j;
};

// wrapper of apply_gate as a function of the gate
static void apply_matchgate_v(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct apply_matchgate_v_params* params = p;

	struct statevector psi_out;
	allocate_statevector(params->psi->nqubits, &psi_out);

	apply_matchgate((struct matchgate*)x, params->i, params->j, params->psi, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << psi_out.nqubits) * sizeof(numeric));

	free_statevector(&psi_out);
}

char* test_apply_matchgate_backward()
{
	int L = 9;

	hid_t file = H5Fopen("../test/data/test_apply_matchgate_backward" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate_backward failed";
	}
	
	struct matchgate V;
	if (read_hdf5_dataset(file, "v/center", H5T_NATIVE_DOUBLE, V.center_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	if (read_hdf5_dataset(file, "v/corner", H5T_NATIVE_DOUBLE, V.corner_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	// input statevector
	struct statevector psi;
	if (allocate_statevector(L, &psi) < 0) { return "memory allocation failed"; }
	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct statevector dpsi_out;
	if (allocate_statevector(L, &dpsi_out) < 0) { return "memory allocation failed"; }
	if (read_hdf5_dataset(file, "dpsi_out", H5T_NATIVE_DOUBLE, dpsi_out.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	const int i_list[3] = { 2, 3, 4 };
	const int j_list[3] = { 5, 4, 1 };

	for (int k = 0; k < 3; k++)
	{
		// backward pass
		struct statevector dpsi;
		if (allocate_statevector(L, &dpsi) < 0) { return "memory allocation failed"; }
		struct matchgate dV;
		apply_matchgate_backward(&V, i_list[k], j_list[k], &psi, &dpsi_out, &dpsi, &dV);

		const double h = 1e-5;

		// numerical gradient with respect to 'psi'
		struct apply_matchgate_psi_params params_psi = {
			.nqubits = L,
			.gate = &V,
			.i = i_list[k],
			.j = j_list[k],
		};
		struct statevector dpsi_num;
		if (allocate_statevector(L, &dpsi_num) < 0) { return "memory allocation failed"; }
		numerical_gradient_backward(apply_matchgate_psi, &params_psi, 1 << L, psi.data, 1 << L, dpsi_out.data, h, dpsi_num.data);

		// compare
		if (uniform_distance((long)1 << L, dpsi.data, dpsi_num.data) > 1e-8) {
			return "gradient of 'apply_gate' with respect to 'psi' does not match finite difference approximation";
		}

		// numerical gradient with respect to the gate
		struct apply_matchgate_v_params params_v = {
			.psi = &psi,
			.i = i_list[k],
			.j = j_list[k],
		};
		struct matchgate dV_num;
		numerical_gradient_backward(apply_matchgate_v, &params_v, 8, (numeric*)&V, 1 << L, dpsi_out.data, h, (numeric*)&dV_num);

		// compare
		if (uniform_distance(8, (numeric*)&dV, (numeric*)&dV_num) > 1e-8) {
				return  "gradient of 'apply_matchgate' with respect to 'V' does not match finite difference approximation";
			}

		free_statevector(&dpsi_num);
		free_statevector(&dpsi);
	}

	free_statevector(&dpsi_out);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


char* test_apply_matchgate_backward_array()
{
	int L = 9;

	hid_t file = H5Fopen("../test/data/test_apply_matchgate_backward_array" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate_backward_array failed";
	}

	struct matchgate V;
	if (read_hdf5_dataset(file, "v/center", H5T_NATIVE_DOUBLE, V.center_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	if (read_hdf5_dataset(file, "v/corner", H5T_NATIVE_DOUBLE, V.corner_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	// input statevector
	struct statevector_array psi;
	if (allocate_statevector_array(L, 8, &psi) < 0) { return "memory allocation failed"; }
	if (read_hdf5_dataset(file, "psi_array", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	int i;

	struct statevector dpsi_out;
	if (allocate_statevector(L, &dpsi_out) < 0) { return "memory allocation failed"; }
	if (read_hdf5_dataset(file, "dpsi_out", H5T_NATIVE_DOUBLE, dpsi_out.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	const int i_list[5] = { 2, 3, 4, 7, 5 };
	const int j_list[5] = { 5, 4, 1, 6, 8 };

	int k;
	for (k = 0; k < 5; k++){

		// backward pass
		struct matchgate dV[8];
		apply_matchgate_backward_array(&V, i_list[k], j_list[k], &psi, &dpsi_out, dV);

		struct statevector dpsi_ref;
		struct matchgate dV_ref;
		if (allocate_statevector(L, &dpsi_ref) < 0) { return "memory allocation failed"; }

		for (i = 0; i < 8; i++){
			char varname[32];
			sprintf(varname, "psi%i", i);
			struct statevector psix;
			if (allocate_statevector(L, &psix) < 0) { return "memory allocation failed"; }
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, psix.data) < 0) {
			return "reading input statevector data from disk failed";
			}

			apply_matchgate_backward(&V, i_list[k], j_list[k], &psix, &dpsi_out, &dpsi_ref, &dV_ref);


			if (uniform_distance(8, (numeric*)&dV[i], (numeric*)&dV_ref) > 1e-8) {
				return "gate gradient of apply_backward_array is not consistent with single backward.";
			}

			free_statevector(&psix);
		}

		free_statevector(&dpsi_ref);
	}

	free_statevector_array(&psi);
	free_statevector(&dpsi_out);

	H5Fclose(file);

	return 0;
}


char* test_apply_matchgate_to_array()
{
	int L = 6;
	int nstates = 5;

	hid_t file = H5Fopen("../test/data/test_apply_matchgate_to_array" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_matchgate_to_array failed";
	}

	struct statevector_array psi, chi1, chi1ref, chi2, chi2ref, chi3, chi3ref;
	if (allocate_statevector_array(L, nstates, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi1)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi1ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi2)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi2ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi3)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi3ref) < 0) { return "memory allocation failed"; }

	struct matchgate V;
	if (read_hdf5_dataset(file, "v/center", H5T_NATIVE_DOUBLE, V.center_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	if (read_hdf5_dataset(file, "v/corner", H5T_NATIVE_DOUBLE, V.corner_block.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi1", H5T_NATIVE_DOUBLE, chi1ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi2", H5T_NATIVE_DOUBLE, chi2ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi3", H5T_NATIVE_DOUBLE, chi3ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}

	// apply the gate
	apply_matchgate_to_array(&V, 2, 5, &psi, &chi1);
	apply_matchgate_to_array(&V, 4, 1, &psi, &chi2);
	apply_matchgate_to_array(&V, 3, 4, &psi, &chi3);

	// compare with reference
	if (uniform_distance(((long)1 << L) * nstates, chi1.data, chi1ref.data) > 1e-12) { return "quantum state array after applying gate does not match reference"; }
	if (uniform_distance(((long)1 << L) * nstates, chi2.data, chi2ref.data) > 1e-12) { return "quantum state array after applying gate does not match reference"; }
	if (uniform_distance(((long)1 << L) * nstates, chi3.data, chi3ref.data) > 1e-12) { return "quantum state array after applying gate does not match reference"; }

	free_statevector_array(&chi3ref);
	free_statevector_array(&chi3);
	free_statevector_array(&chi2ref);
	free_statevector_array(&chi2);
	free_statevector_array(&chi1ref);
	free_statevector_array(&chi1);
	free_statevector_array(&psi);

	H5Fclose(file);

	return 0;
}


char* test_apply_matchgate_placeholder()
{
	int L = 7;

	hid_t file = H5Fopen("../test/data/test_apply_matchgate_placeholder" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_matchgate_placeholder failed";
	}

	struct statevector psi;
	if (allocate_statevector(L, &psi) < 0) { return "memory allocation failed"; }
	struct statevector_array psi_out, psi_out2, psi_out_ref;
	if (allocate_statevector_array(L, 8, &psi_out)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, 16, &psi_out2)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, 16, &psi_out_ref) < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	// case i < j
	if (read_hdf5_dataset(file, "psi_out1", H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
		return "reading output statevector array data from disk failed";
	}
	// apply gate placeholde
	apply_matchgate_placeholder(2, 5, &psi, &psi_out);
	apply_gate_placeholder(2, 5, &psi, &psi_out2);
	// compare with reference
	if (uniform_distance(((long)1 << L)*16, psi_out2.data, psi_out_ref.data) > 1e-12) {
		return "quantum state array after applying gate placeholder does not match reference";
	}

	int map[8] = {5, 6, 9, 10, 0, 3, 12, 15};
	int st;
	int ite;
	for (st = 0; st < 8; st++){
		for (ite = 0; ite < (long)1 << L; ite++){
			if (uniform_distance(1, &psi_out.data[ite*8 + st], &psi_out2.data[ite*16 + map[st]]) > 1e-12) {
				return "1 single quantum state array after applying matchgate placeholder does not match reference";
			}	
		}
	}

	// case i > j
	if (read_hdf5_dataset(file, "psi_out2", H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
		return "reading output statevector array data from disk failed";
	}
	// apply gate placeholder
	apply_matchgate_placeholder(5, 1, &psi, &psi_out);
	apply_gate_placeholder(5, 1, &psi, &psi_out2);
	// compare with reference
	if (uniform_distance(((long)1 << L)*16, psi_out2.data, psi_out_ref.data) > 1e-12) {
		return "quantum state array after applying gate placeholder does not match reference";
	}

	for (st = 0; st < 8; st++){
		for (ite = 0; ite < (long)1 << L; ite++){
			if (uniform_distance(1, &psi_out.data[ite*8 + st], &psi_out2.data[ite*16 + map[st]]) > 1e-12) {
				return "1 single quantum state array after applying matchgate placeholder does not match reference";
			}	
		}
	}

	free_statevector_array(&psi_out_ref);
	free_statevector_array(&psi_out);
	free_statevector_array(&psi_out2);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include "matchgate_target.h"
#include "matchgate_brickwall.h"
#include "matchgate.h"

#ifdef MPI
	#include <mpi.h>
#endif


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} C],
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state.
///
int matchgate_circuit_unitary_target(linear_func ufunc, void* udata, const struct matchgate gates[], const int ngates, const int wires[], const int nqubits, numeric* fval)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Cpsi = { 0 };
	if (allocate_statevector(nqubits, &Cpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		int ret;

		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		ret = apply_quantum_matchgate_circuit(gates, ngates, wires, &psi, &Cpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'apply_quantum_circuit' failed, return value: %i\n", ret);
			return -1;
		}

		// f += <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Cpsi.data[a];
		}
	}

	free_statevector(&Cpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


int parallel_matchgate_circuit_unitary_target(linear_func ufunc, void* udata, const struct matchgate gates[], const int ngates, const int wires[], const int nqubits, numeric* fval)
{
	const intqs n = (intqs)1 << nqubits;
	numeric f_total = 0;

	#ifdef REMOVE_CRITICAL
	#pragma omp parallel
	{
		// temporary statevectors
		struct statevector psi = { 0 };
		if (allocate_statevector(nqubits, &psi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Upsi = { 0 };
		if (allocate_statevector(nqubits, &Upsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Cpsi = { 0 };
		if (allocate_statevector(nqubits, &Cpsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}

		numeric f_partial = 0;
		// implement trace via summation over unit vectors
		#pragma omp for 
		for (intqs b = 0; b < n; b++)
		{
			int ret;

			memset(psi.data, 0, n * sizeof(numeric));
			psi.data[b] = 1;

			ret = ufunc(&psi, udata, &Upsi);
			if (ret < 0) {
				fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			}
			// negate and complex-conjugate entries
			for (intqs a = 0; a < n; a++)
			{
				Upsi.data[a] = -conj(Upsi.data[a]);
			}

			ret = apply_quantum_matchgate_circuit(gates, ngates, wires, &psi, &Cpsi);
			if (ret < 0) {
				fprintf(stderr, "call of 'apply_quantum_circuit' failed, return value: %i\n", ret);
			}

			// f += <Upsi | Wpsi>
			for (intqs a = 0; a < n; a++)
			{
				f_partial += Upsi.data[a] * Cpsi.data[a];
			}
		}

		free_statevector(&Cpsi);
		free_statevector(&Upsi);
		free_statevector(&psi);	
	}
	f_total = 1;
	#else 

	#pragma omp parallel reduction(+:f_total)
	{
		// temporary statevectors
		struct statevector psi = { 0 };
		if (allocate_statevector(nqubits, &psi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Upsi = { 0 };
		if (allocate_statevector(nqubits, &Upsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Cpsi = { 0 };
		if (allocate_statevector(nqubits, &Cpsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}

		numeric f_partial = 0;
		// implement trace via summation over unit vectors
		#pragma omp for 
		for (intqs b = 0; b < n; b++)
		{
			int ret;

			memset(psi.data, 0, n * sizeof(numeric));
			psi.data[b] = 1;

			ret = ufunc(&psi, udata, &Upsi);
			if (ret < 0) {
				fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			}
			// negate and complex-conjugate entries
			for (intqs a = 0; a < n; a++)
			{
				Upsi.data[a] = -conj(Upsi.data[a]);
			}

			ret = apply_quantum_matchgate_circuit(gates, ngates, wires, &psi, &Cpsi);
			if (ret < 0) {
				fprintf(stderr, "call of 'apply_quantum_circuit' failed, return value: %i\n", ret);
			}

			// f += <Upsi | Wpsi>
			for (intqs a = 0; a < n; a++)
			{
				f_partial += Upsi.data[a] * Cpsi.data[a];
			}
		}

		f_total += f_partial;

		free_statevector(&Cpsi);
		free_statevector(&Upsi);
		free_statevector(&psi);	
	}

	#endif
	*fval = f_total;	

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} C] and its gate gradients,
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state.
///
int matchgate_circuit_unitary_target_and_gradient(linear_func ufunc, void* udata, const struct matchgate gates[], const int ngates, const int wires[], const int nqubits, numeric* fval, struct matchgate dgates[])
{
	// temporary statevectors
	struct statevector psi;
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi;
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Cpsi;
	if (allocate_statevector(nqubits, &Cpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache;
	if (allocate_quantum_circuit_cache(nqubits, ngates, &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct matchgate* dgates_unit = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct matchgate));
	if (dgates_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", ngates);
		return -1;
	}

	for (int i = 0; i < ngates; i++)
	{
		memset(dgates[i].center_block.data, 0, sizeof(dgates[i].center_block.data));
		memset(dgates[i].corner_block.data, 0, sizeof(dgates[i].corner_block.data));
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		int ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		// quantum circuit forward pass
		if (quantum_matchgate_circuit_forward(gates, ngates, wires, &psi, &cache, &Cpsi) < 0) {
			fprintf(stderr, "'quantum_circuit_forward' failed internally");
			return -3;
		}

		// f += <Upsi | Cpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Cpsi.data[a];
		}

		// quantum circuit backward pass
		// note: overwriting 'psi' with gradient
		if (quantum_matchgate_circuit_backward(gates, ngates, wires, &cache, &Upsi, &psi, dgates_unit) < 0) {
			fprintf(stderr, "'quantum_circuit_backward' failed internally");
			return -4;
		}
		// accumulate gate gradients for current unit vector
		for (int i = 0; i < ngates; i++)
		{
			add_matchgate(&dgates[i], &dgates_unit[i]);
		}
	}

	aligned_free(dgates_unit);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Cpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}

//________________________________________________________________________________________________________________________
///
/// \brief Convert a matchgate brickwall to a sequential circuit.
///
static inline void matchgate_brickwall_to_sequential(const int nqubits, const int nlayers, const struct matchgate* restrict vlist, const int* perms[], struct matchgate* restrict gates, int* restrict wires)
{
	assert(nqubits % 2 == 0);
	const int ngateslayer = nqubits / 2;

	for (int i = 0; i < nlayers; i++)
	{
		for (int j = 0; j < ngateslayer; j++)
		{
			// duplicate gate vlist[i]
			memcpy(gates[i*ngateslayer + j].center_block.data, vlist[i].center_block.data, sizeof(vlist[i].center_block.data));
			memcpy(gates[i*ngateslayer + j].corner_block.data, vlist[i].corner_block.data, sizeof(vlist[i].corner_block.data));
			wires[2 * (i*ngateslayer + j)    ] = perms[i][2*j    ];
			wires[2 * (i*ngateslayer + j) + 1] = perms[i][2*j + 1];
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W],
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int matchgate_brickwall_unitary_target(linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval)
{
	const int ngates = nlayers * (nqubits / 2);

	struct matchgate* gates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct matchgate));
	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	matchgate_brickwall_to_sequential(nqubits, nlayers, vlist, perms, gates, wires);

	#ifdef STATEVECTOR_PARALLELIZATION
	int ret = parallel_matchgate_circuit_unitary_target(ufunc, udata, gates, ngates, wires, nqubits, fval);
	# else
	int ret = matchgate_circuit_unitary_target(ufunc, udata, gates, ngates, wires, nqubits, fval);
	#endif
	aligned_free(wires);
	aligned_free(gates);

	return ret;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W] and its gate gradients,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int matchgate_brickwall_unitary_target_and_gradient(linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval, struct matchgate dvlist[])
{
	const int ngates = nlayers * (nqubits / 2);

	struct matchgate* gates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct matchgate));
	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	matchgate_brickwall_to_sequential(nqubits, nlayers, vlist, perms, gates, wires);

	struct matchgate* dgates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct matchgate));
	int ret = matchgate_circuit_unitary_target_and_gradient(ufunc, udata, gates, ngates, wires, nqubits, fval, dgates);

	// accumulate gradients
	const int ngateslayer = nqubits / 2;
	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));

		for (int j = 0; j < ngateslayer; j++)
		{
			add_matchgate(&dvlist[i], &dgates[i*ngateslayer + j]);
		}
	}

	aligned_free(dgates);
	aligned_free(wires);
	aligned_free(gates);

	return ret;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W], its gate gradients and Hessian,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int matchgate_brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval, struct matchgate dvlist[], numeric* hess)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(nqubits, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache = { 0 };
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct matchgate* dvlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (dvlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));
	}

	const int m = nlayers * 8;
	memset(hess, 0, m * m * sizeof(numeric));

	numeric* hess_unit = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	if (hess_unit == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		int ret;

		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		// brickwall unitary forward pass
		if (matchgate_brickwall_unitary_forward(vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}
		
		// f += <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Wpsi.data[a];
		}
		
		// brickwall unitary backward pass and Hessian computation
		// note: overwriting 'psi' with gradient
		#ifdef TRANSLATIONAL_INVARIANCE
		if (matchgate_ti_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			return -4;
		}
		#else
		if (matchgate_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			return -4;
		}
		#endif
		
		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matchgate(&dvlist[i], &dvlist_unit[i]);
		}

		// accumulate Hessian matrix for current unit vector
		for (int i = 0; i < m*m; i++)
		{
			hess[i] += hess_unit[i];
		}
	}

	aligned_free(hess_unit);
	aligned_free(dvlist_unit);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}

int matchgate_ti_brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval, struct matchgate dvlist[], numeric* hess)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(nqubits, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache = { 0 };
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct matchgate* dvlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (dvlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));
	}

	const int m = nlayers * 8;
	memset(hess, 0, m * m * sizeof(numeric));

	numeric* hess_unit = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	if (hess_unit == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		int ret;

		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		// brickwall unitary forward pass
		if (matchgate_brickwall_unitary_forward(vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}
		
		// f += <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Wpsi.data[a];
		}
		
		// brickwall unitary backward pass and Hessian computation
		// note: overwriting 'psi' with gradient
		if (matchgate_ti_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			return -4;
		}
		
		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matchgate(&dvlist[i], &dvlist_unit[i]);
		}

		// accumulate Hessian matrix for current unit vector
		for (int i = 0; i < m*m; i++)
		{
			hess[i] += hess_unit[i];
		}
	}

	aligned_free(hess_unit);
	aligned_free(dvlist_unit);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}

int parallel_matchgate_brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval, struct matchgate dvlist[], numeric* hess)
{	
	for (int i = 0; i < nlayers; i++)
		{
			memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
			memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));
		}

	const int m = nlayers * 8;
	memset(hess, 0, m * m * sizeof(numeric));

	const intqs n = (intqs)1 << nqubits;
	numeric f_total = 0;

	#ifdef REMOVE_CRITICAL

	#pragma omp parallel
	{
		struct statevector psi = { 0 };
		if (allocate_statevector(nqubits, &psi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Upsi = { 0 };
		if (allocate_statevector(nqubits, &Upsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Wpsi = { 0 };
		if (allocate_statevector(nqubits, &Wpsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}

		struct quantum_circuit_cache cache = { 0 };
		if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
			fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		}

		struct matchgate* dvlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
		struct matchgate* dvlist_partial = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
		if (dvlist_unit == NULL || dvlist_partial == NULL)  {
			fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		}

		numeric* hess_unit = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		numeric* hess_partial = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (hess_unit == NULL || hess_partial == NULL) {
			fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		}

		for (int i = 0; i < nlayers; i++)
		{
			memset(dvlist_partial[i].center_block.data, 0, sizeof(dvlist_partial[i].center_block.data));
			memset(dvlist_partial[i].corner_block.data, 0, sizeof(dvlist_partial[i].corner_block.data));
		}

		memset(hess_partial, 0, m * m * sizeof(numeric));
		
		numeric f_partial = 0;
		// implement trace via parallel summation over unit vectors
		#pragma omp for
		for (intqs b = 0; b < n; b++)
		{	
			int ret;
			memset(psi.data, 0, n * sizeof(numeric));
			psi.data[b] = 1;
			
			ret = ufunc(&psi, udata, &Upsi);

			if(ret < 0){
				fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			}
			
			// negate and complex-conjugate entries
			for (intqs a = 0; a < n; a++)
			{
				Upsi.data[a] = -conj(Upsi.data[a]);
			}

			// brickwall unitary forward pass
			matchgate_brickwall_unitary_forward(vlist, nlayers, perms, &psi, &cache, &Wpsi);
			
			
			// f += <Upsi | Wpsi>
			for (intqs a = 0; a < n; a++)
			{
				f_partial += Upsi.data[a] * Wpsi.data[a];
			}
			
			// brickwall unitary backward pass and Hessian computation
			// note: overwriting 'psi' with gradient
			#ifdef TRANSLATIONAL_INVARIANCE 
			matchgate_ti_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit);
			#else
			matchgate_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit);
			#endif
			
			// accumulate gate gradients for current thread
			for (int i = 0; i < nlayers; i++)
			{
				add_matchgate(&dvlist_partial[i], &dvlist_unit[i]);
			}

			// accumulate Hessian matrix for current thread
			for (int i = 0; i < m*m; i++)
			{
				hess_partial[i] += hess_unit[i];
			}
		}

		aligned_free(hess_unit);
		aligned_free(dvlist_unit);
		aligned_free(hess_partial);
		aligned_free(dvlist_partial);
		free_quantum_circuit_cache(&cache);
		free_statevector(&Wpsi);
		free_statevector(&Upsi);
		free_statevector(&psi);
	}
	f_total = 1;
	#else 

	#pragma omp parallel reduction(+:f_total)
	{
		struct statevector psi = { 0 };
		if (allocate_statevector(nqubits, &psi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Upsi = { 0 };
		if (allocate_statevector(nqubits, &Upsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Wpsi = { 0 };
		if (allocate_statevector(nqubits, &Wpsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}

		struct quantum_circuit_cache cache = { 0 };
		if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
			fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		}

		struct matchgate* dvlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
		struct matchgate* dvlist_partial = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
		if (dvlist_unit == NULL || dvlist_partial == NULL)  {
			fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		}

		numeric* hess_unit = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		numeric* hess_partial = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (hess_unit == NULL || hess_partial == NULL) {
			fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		}

		for (int i = 0; i < nlayers; i++)
		{
			memset(dvlist_partial[i].center_block.data, 0, sizeof(dvlist_partial[i].center_block.data));
			memset(dvlist_partial[i].corner_block.data, 0, sizeof(dvlist_partial[i].corner_block.data));
		}

		memset(hess_partial, 0, m * m * sizeof(numeric));
		
		numeric f_partial = 0;
		// implement trace via parallel summation over unit vectors
		#pragma omp for
		for (intqs b = 0; b < n; b++)
		{	
			int ret;
			memset(psi.data, 0, n * sizeof(numeric));
			psi.data[b] = 1;
			
			ret = ufunc(&psi, udata, &Upsi);

			if(ret < 0){
				fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			}
			
			// negate and complex-conjugate entries
			for (intqs a = 0; a < n; a++)
			{
				Upsi.data[a] = -conj(Upsi.data[a]);
			}

			// brickwall unitary forward pass
			matchgate_brickwall_unitary_forward(vlist, nlayers, perms, &psi, &cache, &Wpsi);
			
			
			// f += <Upsi | Wpsi>
			for (intqs a = 0; a < n; a++)
			{
				f_partial += Upsi.data[a] * Wpsi.data[a];
			}
			
			// brickwall unitary backward pass and Hessian computation
			// note: overwriting 'psi' with gradient
			#ifdef TRANSLATIONAL_INVARIANCE 
			matchgate_ti_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit);
			#else
			matchgate_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit);
			#endif
			
			// accumulate gate gradients for current thread
			for (int i = 0; i < nlayers; i++)
			{
				add_matchgate(&dvlist_partial[i], &dvlist_unit[i]);
			}

			// accumulate Hessian matrix for current thread
			for (int i = 0; i < m*m; i++)
			{
				hess_partial[i] += hess_unit[i];
			}
			
		}

		#pragma omp critical
		{	
			// accumulate gate gradients from each thread
			for (int i = 0; i < nlayers; i++)
			{
				add_matchgate(&dvlist[i], &dvlist_partial[i]);
			}

			// accumulate Hessian matrix from each thread
			for (int i = 0; i < m*m; i++)
			{
				hess[i] += hess_partial[i];
			}
		}

		f_total += f_partial;

		aligned_free(hess_unit);
		aligned_free(dvlist_unit);
		aligned_free(hess_partial);
		aligned_free(dvlist_partial);
		free_quantum_circuit_cache(&cache);
		free_statevector(&Wpsi);
		free_statevector(&Upsi);
		free_statevector(&psi);
	}
	#endif
	*fval = f_total;

	return 0;
}

//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W], its gate gradient as real vector and Hessian matrix,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix(linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval, double* grad_vec, double* H)
{
	struct matchgate* dvlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	if (dvlist == NULL)
	{
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	const int m = nlayers * 8;
	numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	
	#ifdef STATEVECTOR_PARALLELIZATION 
	int ret = parallel_matchgate_brickwall_unitary_target_gradient_hessian(ufunc, udata, vlist, nlayers, nqubits, perms, fval, dvlist, hess);
	if (ret < 0) {
		return ret;
	}
	#else
	int ret = matchgate_brickwall_unitary_target_gradient_hessian(ufunc, udata, vlist, nlayers, nqubits, perms, fval, dvlist, hess);
	if (ret < 0) {
		return ret;
	}
	#endif


	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matchgate(&dvlist[i]);
		tangent_matchgate_to_real(&vlist[i], &dvlist[i], &grad_vec[i * 8]);
	}

	// project blocks of Hessian matrix
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = i; j < nlayers; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				// unit vector
				double r[8] = { 0 };
				r[k] = 1;
				struct matchgate Z;
				real_to_tangent_matchgate(r, &vlist[j], &Z);

				// could use zgemv for matrix vector multiplication, but not performance critical
				struct matchgate G = { 0 };
				for (int x = 0; x < 4; x++) {
					for (int y = 0; y < 4; y++) {
						G.center_block.data[x] += hess[((i*8 + x)*nlayers + j)*8 + y    ] * Z.center_block.data[y]
								   				+ hess[((i*8 + x)*nlayers + j)*8 + y + 4] * Z.corner_block.data[y];

						G.corner_block.data[x] += hess[((i*8 + x + 4)*nlayers + j)*8 + y    ] * Z.center_block.data[y]
								                + hess[((i*8 + x + 4)*nlayers + j)*8 + y + 4] * Z.corner_block.data[y];
					}
				}
				// conjugation due to convention for derivative
				conjugate_matchgate(&G);
				
				if (i == j)
				{
					struct matchgate Gproj;
					project_tangent_matchgate(&vlist[i], &G, &Gproj);
					memcpy(G.center_block.data, Gproj.center_block.data, sizeof(G.center_block.data));
					memcpy(G.corner_block.data, Gproj.corner_block.data, sizeof(G.corner_block.data));
					// additional terms resulting from the projection of the gradient
					// onto the Stiefel manifold (unitary matrices)
					struct matchgate gradh;
					adjoint_matchgate(&dvlist[i], &gradh);
					// G -= 0.5 * (Z @ grad^{\dagger} @ V + V @ grad^{\dagger} @ Z)
					struct matchgate T;
					symmetric_triple_matchgate_product(&Z, &gradh, &vlist[i], &T);
					sub_matchgate(&G, &T);
				}

				// represent tangent vector of Stiefel manifold at vlist[i] as real vector
				tangent_matchgate_to_real(&vlist[i], &G, r);
				for (int x = 0; x < 8; x++) {
					H[((i*8 + x)*nlayers + j)*8 + k] = r[x];
				}
			}
		}
	}

	// copy upper triangular part according to symmetry
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < i; j++) {
			H[i*m + j] = H[j*m + i];
		}
	}

	aligned_free(hess);
	aligned_free(dvlist);

	return 0;
}



// MPI implementation

#ifdef MPI

int mpi_parallel_matchgate_circuit_unitary_target(linear_func ufunc, void* udata, const struct matchgate gates[], const int ngates, const int wires[],
    int start, int end, const int nqubits, numeric* fval)
{
	const intqs n = (intqs)1 << nqubits;
	numeric f_total = 0;

	#pragma omp parallel reduction(+:f_total)
	{
		// temporary statevectors
		struct statevector psi = { 0 };
		if (allocate_statevector(nqubits, &psi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Upsi = { 0 };
		if (allocate_statevector(nqubits, &Upsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Cpsi = { 0 };
		if (allocate_statevector(nqubits, &Cpsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}

		numeric f_partial = 0;
		// implement trace via summation over unit vectors
		#pragma omp for 
		for (intqs b = start; b < end; b++)
		{
			int ret;

			memset(psi.data, 0, n * sizeof(numeric));
			psi.data[b] = 1;

			ret = ufunc(&psi, udata, &Upsi);
			if (ret < 0) {
				fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			}
			// negate and complex-conjugate entries
			for (intqs a = 0; a < n; a++)
			{
				Upsi.data[a] = -conj(Upsi.data[a]);
			}

			ret = apply_quantum_matchgate_circuit(gates, ngates, wires, &psi, &Cpsi);
			if (ret < 0) {
				fprintf(stderr, "call of 'apply_quantum_circuit' failed, return value: %i\n", ret);
			}

			// f += <Upsi | Wpsi>
			for (intqs a = 0; a < n; a++)
			{
				f_partial += Upsi.data[a] * Cpsi.data[a];
			}
		}

		f_total += f_partial;

		free_statevector(&Cpsi);
		free_statevector(&Upsi);
		free_statevector(&psi);	
	}
	*fval = f_total;	

	return 0;
}


int mpi_matchgate_brickwall_unitary_target(linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], 
int start, int end, numeric* fval)
{
	const int ngates = nlayers * (nqubits / 2);

	struct matchgate* gates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct matchgate));
	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	matchgate_brickwall_to_sequential(nqubits, nlayers, vlist, perms, gates, wires);

	int ret = mpi_parallel_matchgate_circuit_unitary_target(ufunc, udata, gates, ngates, wires, start, end, nqubits, fval);
	
	aligned_free(wires);
	aligned_free(gates);

	return ret;
}


int mpi_parallel_matchgate_brickwall_unitary_target_gradient_hessian(const int start, const int end, linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, 
const int nqubits, const int* perms[], numeric* fval, struct matchgate dvlist[], numeric* hess)
{	
	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));
	}

	const int m = nlayers * 8;
	memset(hess, 0, m * m * sizeof(numeric));

	numeric f_total = 0;

	#pragma omp parallel reduction(+:f_total)
	{
	// temporary statevectors
		struct statevector psi = { 0 };
		if (allocate_statevector(nqubits, &psi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Upsi = { 0 };
		if (allocate_statevector(nqubits, &Upsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}
		struct statevector Wpsi = { 0 };
		if (allocate_statevector(nqubits, &Wpsi) < 0) {
			fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		}

		struct quantum_circuit_cache cache = { 0 };
		if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
			fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		}

		struct matchgate* dvlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
		struct matchgate* dvlist_partial = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
		if (dvlist_unit == NULL || dvlist_partial == NULL)  {
			fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		}
		
		numeric* hess_unit = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		numeric* hess_partial = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (hess_unit == NULL || hess_partial == NULL) {
			fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		}

		for (int i = 0; i < nlayers; i++)
		{
			memset(dvlist_partial[i].center_block.data, 0, sizeof(dvlist_partial[i].center_block.data));
			memset(dvlist_partial[i].corner_block.data, 0, sizeof(dvlist_partial[i].corner_block.data));
		}

		memset(hess_partial, 0, m * m * sizeof(numeric));

		numeric f_partial = 0;
		// implement trace via summation over unit vectors
		const intqs n = (intqs)1 << nqubits;

		#pragma omp for 
		for (intqs b = start; b < end; b++)
		{
			int ret;

			memset(psi.data, 0, n * sizeof(numeric));
			psi.data[b] = 1;

			ret = ufunc(&psi, udata, &Upsi);
			if (ret < 0) {
				fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			}
			// negate and complex-conjugate entries
			for (intqs a = 0; a < n; a++)
			{
				Upsi.data[a] = -conj(Upsi.data[a]);
			}
		
			// brickwall unitary forward pass
			if (matchgate_brickwall_unitary_forward(vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
				fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			}
			
			// f += <Upsi | Wpsi>
			for (intqs a = 0; a < n; a++)
			{
				f_partial += Upsi.data[a] * Wpsi.data[a];
			}
			
			// brickwall unitary backward pass and Hessian computation
			// note: overwriting 'psi' with gradient
			#ifdef TRANSLATIONAL_INVARIANCE
			if (matchgate_ti_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit) < 0) {
				fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			}
			#else

			if (matchgate_brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit) < 0) {
				fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			}
			#endif
			
			// accumulate gate gradients for current thread
			for (int i = 0; i < nlayers; i++)
			{
				add_matchgate(&dvlist_partial[i], &dvlist_unit[i]);
			}

			// accumulate Hessian matrix for current thread
			for (int i = 0; i < m*m; i++)
			{
				hess_partial[i] += hess_unit[i];
			}
				
		}

		#pragma omp critical
		{	
			// accumulate gate gradients from each thread
			for (int i = 0; i < nlayers; i++)
			{
				add_matchgate(&dvlist[i], &dvlist_partial[i]);
			}

			// accumulate Hessian matrix from each thread
			// TODO: symetrix hessian can be ussed to resduce the loop
			for (int i = 0; i < m*m; i++)
			{
				hess[i] += hess_partial[i];
			}
		}
		
		f_total += f_partial;
		
		aligned_free(hess_unit);
		aligned_free(dvlist_unit);
		aligned_free(hess_partial);
		aligned_free(dvlist_partial);
		free_quantum_circuit_cache(&cache);
		free_statevector(&Wpsi);
		free_statevector(&Upsi);
		free_statevector(&psi);
	}

	*fval = f_total;
	return 0;
}


int mpi_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix(const struct matchgate vlist[], const int nlayers, 
	struct matchgate* dvlist, numeric* hess, double* grad_vec, double* H)
{	

	int m = 8*nlayers;
	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matchgate(&dvlist[i]);
		tangent_matchgate_to_real(&vlist[i], &dvlist[i], &grad_vec[i * 8]);
	}

	// project blocks of Hessian matrix
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = i; j < nlayers; j++)
		{
			for (int k = 0; k < 8; k++)
			{
				// unit vector
				double r[8] = { 0 };
				r[k] = 1;
				struct matchgate Z;
				real_to_tangent_matchgate(r, &vlist[j], &Z);

				// could use zgemv for matrix vector multiplication, but not performance critical
				struct matchgate G = { 0 };
				for (int x = 0; x < 4; x++) {
					for (int y = 0; y < 4; y++) {
						G.center_block.data[x] += hess[((i*8 + x)*nlayers + j)*8 + y    ] * Z.center_block.data[y]
												+ hess[((i*8 + x)*nlayers + j)*8 + y + 4] * Z.corner_block.data[y];

						G.corner_block.data[x] += hess[((i*8 + x + 4)*nlayers + j)*8 + y    ] * Z.center_block.data[y]
												+ hess[((i*8 + x + 4)*nlayers + j)*8 + y + 4] * Z.corner_block.data[y];
					}
				}
				// conjugation due to convention for derivative
				conjugate_matchgate(&G);
				
				if (i == j)
				{
					struct matchgate Gproj;
					project_tangent_matchgate(&vlist[i], &G, &Gproj);
					memcpy(G.center_block.data, Gproj.center_block.data, sizeof(G.center_block.data));
					memcpy(G.corner_block.data, Gproj.corner_block.data, sizeof(G.corner_block.data));
					// additional terms resulting from the projection of the gradient
					// onto the Stiefel manifold (unitary matrices)
					struct matchgate gradh;
					adjoint_matchgate(&dvlist[i], &gradh);
					// G -= 0.5 * (Z @ grad^{\dagger} @ V + V @ grad^{\dagger} @ Z)
					struct matchgate T;
					symmetric_triple_matchgate_product(&Z, &gradh, &vlist[i], &T);
					sub_matchgate(&G, &T);
				}

				// represent tangent vector of Stiefel manifold at vlist[i] as real vector
				tangent_matchgate_to_real(&vlist[i], &G, r);
				for (int x = 0; x < 8; x++) {
					H[((i*8 + x)*nlayers + j)*8 + k] = r[x];
				}
			}
		}
	}

	// copy upper triangular part according to symmetry
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < i; j++) {
			H[i*m + j] = H[j*m + i];
		}
	}

	return 0;
}


int mpi_brickwall_f(int start, int end, linear_func ufunc, void* udata, const struct matchgate* vlist, const int nlayers, const int nqubits,
    const int* perms[], numeric* f_val)
{
    numeric f_partial = 0;
    numeric f_total = 0;

    mpi_matchgate_brickwall_unitary_target(ufunc, udata, vlist, nlayers, nqubits, perms, start, end, &f_partial);

    MPI_Reduce((void*)&f_partial, (void*)&f_total, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    *f_val = f_total;

	return 0;
}


int mpi_gradient_vector_hessian_matrix(int rank, int start, int end, linear_func ufunc, void* udata, const struct matchgate* vlist, const int nlayers, const int nqubits,
    const int* perms[], numeric* f_val, double* grad_vec, double* H){


    if (rank == 0){
        if (grad_vec == NULL || H == NULL){
            printf("NULL pointer\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    const int m = nlayers * 8;

    numeric f_partial = 0;
    struct matchgate* dvlist_partial = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	numeric* hess_partial = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));

    numeric f_total = 0;
    struct matchgate* dvlist_total = NULL;
	numeric* hess_total = NULL;

    if(rank == 0){
        dvlist_total = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct matchgate));
	    hess_total   = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
    }

	
    mpi_parallel_matchgate_brickwall_unitary_target_gradient_hessian(start, end, ufunc, udata, vlist, nlayers, nqubits, perms, 
        &f_partial, dvlist_partial, hess_partial);
		
	
    MPI_Reduce((void*)&f_partial, (void*)&f_total, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((void*)dvlist_partial, (void*)dvlist_total, m, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((void*)hess_partial, (void*)hess_total, m * m, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
	
    if (rank == 0){
        mpi_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix(vlist, nlayers, dvlist_total, hess_total, grad_vec, H);
    }
	
    *f_val = f_total;
	
    aligned_free(dvlist_partial);
    if (rank == 0) {
        aligned_free(dvlist_total);
        aligned_free(hess_total);
    }
    aligned_free(hess_partial);
    return 0;
}

#endif
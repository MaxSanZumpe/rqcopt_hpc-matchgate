#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "matchgate_brickwall.h"
#include "gate.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brickwall-type
/// quantum circuit with periodic boundary conditions to state psi.
///
int apply_matchgate_brickwall_unitary(const struct matchgate vlist[], const int nlayers, const int* perms[], const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);

	const int nstates = nlayers * (psi->nqubits / 2);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(psi->nqubits, &tmp) < 0) {
		fprintf(stderr, "allocating temporary statevector with %i qubits failed\n", psi->nqubits);
		return -1;
	}

	int k = 0;
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = 0; j < psi->nqubits; j += 2)
		{
			const struct statevector* psi0 = (k == 0 ? psi : ((nstates - k) % 2 == 0 ? psi_out : &tmp));
			struct statevector* psi1 = ((nstates - k) % 2 == 0 ? &tmp : psi_out);
			apply_matchgate(&vlist[i], perms[i][j], perms[i][j + 1], psi0, psi1);
			k++;
		}
	}

	free_statevector(&tmp);

	return 0;
}



//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
int matchgate_brickwall_unitary_forward(const struct matchgate vlist[], int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct quantum_circuit_cache* cache, struct statevector* restrict psi_out)
{
	const int nstates = nlayers * (psi->nqubits / 2);

	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);
	assert(cache->ngates == nstates);
	assert(cache->nqubits == psi->nqubits);
	assert(psi->nqubits % 2 == 0);

	// store initial statevector in cache as well
	memcpy(cache->psi_list[0].data, psi->data, ((size_t)1 << psi->nqubits) * sizeof(numeric));

	int k = 0;
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = 0; j < psi->nqubits; j += 2)
		{
			apply_matchgate(&vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k],
				(k + 1 < nstates ? &cache->psi_list[k + 1] : psi_out));
			k++;
		}
	}

	return 0;
}



//________________________________________________________________________________________________________________________
///
/// \brief Backward pass for applying the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
int matchgate_brickwall_unitary_backward(const struct matchgate vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dvlist[])
{
	const int nstates = nlayers * (dpsi->nqubits / 2);

	assert(nlayers >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(dpsi_out->nqubits % 2 == 0);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->ngates == nstates);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(dpsi->nqubits, &tmp) < 0) {
		return -1;
	}

	int k = nstates - 1;
	for (int i = nlayers - 1; i >= 0; i--)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));

		for (int j = dpsi->nqubits - 2; j >= 0; j -= 2)
		{
			const struct statevector* dpsi1 = ((k == nstates - 1) ? dpsi_out : (k % 2 == 1 ? dpsi : &tmp));
			struct statevector* dpsi0 = (k % 2 == 1 ? &tmp : dpsi);

			struct matchgate dV;
			apply_matchgate_backward(&vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k], dpsi1, dpsi0, &dV);
			// accumulate gradient
			add_matchgate(&dvlist[i], &dV);

			k--;
		}
	}

	free_statevector(&tmp);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass and Hessian computation for applying the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
/// On input, 'hess' must point to a memory block of size (nlayers * 8)^2.
///
int matchgate_brickwall_unitary_backward_hessian(const struct matchgate vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dvlist[], numeric* hess)
{
const int nstates = nlayers * (dpsi->nqubits / 2);

	assert(nlayers >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(dpsi_out->nqubits % 2 == 0);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->ngates == nstates);

	// store gradient vectors in another cache
	struct quantum_circuit_cache grad_cache;
	if (allocate_quantum_circuit_cache(dpsi->nqubits, nstates, &grad_cache) < 0) {
		fprintf(stderr, "allocating a brick wall unitary cache failed\n");
		return -1;
	}

	// store initial gradient statevector in cache
	memcpy(grad_cache.psi_list[nstates - 1].data, dpsi_out->data, ((size_t)1 << dpsi_out->nqubits) * sizeof(numeric));

	// gradient
	int k = nstates - 1;
	for (int i = nlayers - 1; i >= 0; i--)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));

		for (int j = dpsi->nqubits - 2; j >= 0; j -= 2)
		{
			struct statevector* dpsi0 = (k > 0 ? &grad_cache.psi_list[k - 1] : dpsi);

			struct matchgate dV;
			apply_matchgate_backward(&vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k], &grad_cache.psi_list[k], dpsi0, &dV);
			// accumulate gradient
			add_matchgate(&dvlist[i], &dV);

			k--;
		}
	}

	// Hessian
	memset(hess, 0, nlayers * 8 * nlayers * 8 * sizeof(numeric));
	// temporary statevector array
	struct statevector_array tmp[2];
	for (int i = 0; i < 2; i++) {
		if (allocate_statevector_array(dpsi->nqubits, 8, &tmp[i]) < 0) {
			fprintf(stderr, "memory allocation of a statevector array for %i qubits and 8 states failed", dpsi->nqubits);
			return -1;
		}
	}
	// temporary derivative with respect to two gates
	struct matchgate* h = aligned_alloc(MEM_DATA_ALIGN, 8 * sizeof(struct matchgate));
	if (h == NULL) {
		fprintf(stderr, "allocating temporary gates failed\n");
		return -1;
	}
	k = 0;
	for (int i = 0; i < nlayers; i++)
	{
		for (int r = 0; r < dpsi->nqubits; r += 2)
		{
			apply_matchgate_placeholder(perms[i][r], perms[i][r + 1], &cache->psi_list[k], &tmp[0]);
			k++;

			// proceed through the circuit with gate placeholder at layer i and qubit pair indexed by r
			int p = 0;
			int l = k;
			for (int j = i; j < nlayers; j++)
			{
				for (int s = 0; s < dpsi->nqubits; s += 2)
				{
					if (i == j && s <= r) {
						continue;
					}

					apply_matchgate_backward_array(&vlist[j], perms[j][s], perms[j][s + 1], &tmp[p], &grad_cache.psi_list[l], h);
					// accumulate Hessian entries
					for (int x = 0; x < 8; x++) {
						for (int y = 0; y < 4; y++) {
							hess[((i*8 + x)*nlayers + j)*8 + y + 0] += h[x].center_block.data[y];
							hess[((i*8 + x)*nlayers + j)*8 + y + 4] += h[x].corner_block.data[y];
						}
					}

					if (j < nlayers - 1 || s < dpsi->nqubits - 2) {  // skip (expensive) gate application at final iteration
						apply_matchgate_to_array(&vlist[j], perms[j][s], perms[j][s + 1], &tmp[p], &tmp[1 - p]);
						p = 1 - p;
					}

					l++;
				}
			}
		}
	}

	// symmetrize diagonal blocks
	for (int i = 0; i < nlayers; i++) {
		for (int x = 0; x < 8; x++) {
			for (int y = 0; y <= x; y++) {
				numeric s = hess[((i*8 + x)*nlayers + i)*8 + y] + hess[((i*8 + y)*nlayers + i)*8 + x];
				hess[((i*8 + x)*nlayers + i)*8 + y] = s;
				hess[((i*8 + y)*nlayers + i)*8 + x] = s;
			}
		}
	}
	// copy off-diagonal blocks according to symmetry
	for (int i = 0; i < nlayers; i++) {
		for (int j = 0; j < i; j++) {
			for (int x = 0; x < 8; x++) {
				for (int y = 0; y < 8; y++) {
					hess[((i*8 + x)*nlayers + j)*8 + y] = hess[((j*8 + y)*nlayers + i)*8 + x];
				}
			}
		}
	}
	
	aligned_free(h);
	free_statevector_array(&tmp[1]);
	free_statevector_array(&tmp[0]);
	free_quantum_circuit_cache(&grad_cache);

	return 0;
}


//-----------------------------------------------------------------------------------------
// Implemention of gradient and hessian exploiting translatonal invariance

int matchgate_ti_brickwall_unitary_backward(const struct matchgate vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dvlist[])
{
	const int nstates = nlayers * (dpsi->nqubits / 2);

	assert(nlayers >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(dpsi_out->nqubits % 2 == 0);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->ngates == nstates);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(dpsi->nqubits, &tmp) < 0) {
		return -1;
	}

	int k = nstates - 1;
	int j = dpsi->nqubits - 2;
	for (int i = nlayers - 1; i >= 0; i--)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));
		
		const struct statevector* dpsi1 = ((k == nstates - 1) ? dpsi_out : (k % 2 == 1 ? dpsi : &tmp));
		struct statevector* dpsi0 = (k % 2 == 1 ? &tmp : dpsi);

		struct matchgate dV;
		apply_matchgate_backward(&vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k], dpsi1, dpsi0, &dV);
		// accumulate gradient
		scale_matchgate(&dV, dpsi->nqubits / 2);
		add_matchgate(&dvlist[i], &dV);

		k -= dpsi->nqubits / 2;
	}

	free_statevector(&tmp);

	return 0;
}


int matchgate_ti_brickwall_unitary_backward_hessian(const struct matchgate vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dvlist[], numeric* hess)
{
	const int nstates = nlayers * (dpsi->nqubits / 2);

	assert(nlayers >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(dpsi_out->nqubits % 2 == 0);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->ngates == nstates);

	// store gradient vectors in another cache
	struct quantum_circuit_cache grad_cache;
	if (allocate_quantum_circuit_cache(dpsi->nqubits, nstates, &grad_cache) < 0) {
		fprintf(stderr, "allocating a brick wall unitary cache failed\n");
		return -1;
	}

	// store initial gradient statevector in cache
	memcpy(grad_cache.psi_list[nstates - 1].data, dpsi_out->data, ((size_t)1 << dpsi_out->nqubits) * sizeof(numeric));

	// gradient
	int k = nstates - 1;
	for (int i = nlayers - 1; i >= 0; i--)
	{
		memset(dvlist[i].center_block.data, 0, sizeof(dvlist[i].center_block.data));
		memset(dvlist[i].corner_block.data, 0, sizeof(dvlist[i].corner_block.data));

		for (int j = dpsi->nqubits - 2; j >= 0; j -= 2)
		{
			struct statevector* dpsi0 = (k > 0 ? &grad_cache.psi_list[k - 1] : dpsi);

			struct matchgate dV;
			apply_matchgate_backward(&vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k], &grad_cache.psi_list[k], dpsi0, &dV);
			// accumulate gradient
			add_matchgate(&dvlist[i], &dV);

			k--;
		}
	}

	// Hessian
	memset(hess, 0, nlayers * 8 * nlayers * 8 * sizeof(numeric));
	// temporary statevector array
	struct statevector_array tmp[2];
	for (int i = 0; i < 2; i++) {
		if (allocate_statevector_array(dpsi->nqubits, 8, &tmp[i]) < 0) {
			fprintf(stderr, "memory allocation of a statevector array for %i qubits and 8 states failed", dpsi->nqubits);
			return -1;
		}
	}
	// temporary derivative with respect to two gates
	struct matchgate* h = aligned_alloc(MEM_DATA_ALIGN, 8 * sizeof(struct matchgate));
	if (h == NULL) {
		fprintf(stderr, "allocating temporary gates failed\n");
		return -1;
	}
	k = 0;
	int r = 0;
	int o;
	double mult;
	for (int i = 0; i < nlayers; i++)
	{
		apply_matchgate_placeholder(perms[i][r], perms[i][r + 1], &cache->psi_list[k], &tmp[0]);
		int l = k + 1;
		k += (int)(dpsi->nqubits)/2;
		// proceed through the circuit with gate placeholder at layer i and qubit pair indexed by r
		int p = 0;
		for (int j = i; j < nlayers; j++)
		{
			o = 1;
			for (int s = 0; s < dpsi->nqubits; s += 2)
			{
				if (i == j && s <= r) {
					continue;
				}

				mult = dpsi->nqubits / 2;
				if (i == j) { mult -= o; }
				o++;
				
				apply_matchgate_backward_array(&vlist[j], perms[j][s], perms[j][s + 1], &tmp[p], &grad_cache.psi_list[l], h);
				// accumulate Hessian entries
				for (int x = 0; x < 8; x++) {
					for (int y = 0; y < 4; y++) {
						hess[((i*8 + x)*nlayers + j)*8 + y + 0] += mult*h[x].center_block.data[y];
						hess[((i*8 + x)*nlayers + j)*8 + y + 4] += mult*h[x].corner_block.data[y];
					}
				}

				if (j < nlayers - 1 || s < dpsi->nqubits - 2) {  // skip (expensive) gate application at final iteration
					apply_matchgate_to_array(&vlist[j], perms[j][s], perms[j][s + 1], &tmp[p], &tmp[1 - p]);
					p = 1 - p;
				}

				l++;
			}
		}	
	}

	// symmetrize diagonal blocks
	for (int i = 0; i < nlayers; i++) {
		for (int x = 0; x < 8; x++) {
			for (int y = 0; y <= x; y++) {
				numeric s = hess[((i*8 + x)*nlayers + i)*8 + y] + hess[((i*8 + y)*nlayers + i)*8 + x];
				hess[((i*8 + x)*nlayers + i)*8 + y] = s;
				hess[((i*8 + y)*nlayers + i)*8 + x] = s;
			}
		}
	}
	// copy off-diagonal blocks according to symmetry
	for (int i = 0; i < nlayers; i++) {
		for (int j = 0; j < i; j++) {
			for (int x = 0; x < 8; x++) {
				for (int y = 0; y < 8; y++) {
					hess[((i*8 + x)*nlayers + j)*8 + y] = hess[((j*8 + y)*nlayers + i)*8 + x];
				}
			}
		}
	}
	
	aligned_free(h);
	free_statevector_array(&tmp[1]);
	free_statevector_array(&tmp[0]);
	free_quantum_circuit_cache(&grad_cache);

	return 0;
}
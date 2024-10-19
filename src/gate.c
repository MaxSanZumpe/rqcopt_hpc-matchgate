#include <memory.h>
#include <assert.h>
#include <omp.h>
#include "gate.h"


#ifdef GATE_PARALLELIZATION

//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit matchgate to qubits 'i' and 'j' of a statevector.
///
void apply_matchgate(const struct matchgate* gate, const int i, const int j, const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { .center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			}
			,.corner_block.data =
			{ 
				gate->corner_block.data[0],gate -> corner_block.data[1],
				gate->corner_block.data[2],gate -> corner_block.data[3],
			}
		};
		// flip i <-> j
		apply_matchgate(&gate_perm, j, i, psi, psi_out);
	}
	else if (j == i + 1)
	{
		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(2)
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				psi_out->data[(a*4    )*n + b] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
				psi_out->data[(a*4 + 1)*n + b] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
				psi_out->data[(a*4 + 2)*n + b] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
				psi_out->data[(a*4 + 3)*n + b] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
			}
		}
	}
	else
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					psi_out->data[(((a*2    )*n + b)*2    )*o + c] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
					psi_out->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass for applying a two-qubit gate to qubits 'i' and 'j' of a statevector.
///
void apply_matchgate_backward(const struct matchgate* gate, const int i, const int j, const struct statevector* restrict psi,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate* dgate)
{
	assert(psi->nqubits == dpsi_out->nqubits);
	assert(psi->nqubits == dpsi->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { 
			.center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			},
			.corner_block.data =
			{ 
				gate->corner_block.data[0], gate->corner_block.data[1],
				gate->corner_block.data[2], gate->corner_block.data[3],
			}
		};

		struct matchgate dgate_perm;

		// flip i <-> j
		apply_matchgate_backward(&gate_perm, j, i, psi, dpsi_out, dpsi, &dgate_perm);

		// undo transposition of first and second qubit wire for gate gradient
		dgate->corner_block.data[0] = dgate_perm.corner_block.data[0]; dgate->corner_block.data[1] = dgate_perm.corner_block.data[1];
		dgate->center_block.data[0] = dgate_perm.center_block.data[3]; dgate->center_block.data[1] = dgate_perm.center_block.data[2];
		dgate->center_block.data[2] = dgate_perm.center_block.data[1]; dgate->center_block.data[3] = dgate_perm.center_block.data[0];
		dgate->corner_block.data[2] = dgate_perm.corner_block.data[2]; dgate->corner_block.data[3] = dgate_perm.corner_block.data[3];
	}
	else if (j == i + 1)
	{
		memset(dgate->center_block.data, 0, sizeof(dgate->center_block.data));
		memset(dgate->corner_block.data, 0, sizeof(dgate->corner_block.data));

		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(2) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				numeric dx = dpsi_out->data[(a*4    )*n + b];
				numeric dy = dpsi_out->data[(a*4 + 1)*n + b];
				numeric dz = dpsi_out->data[(a*4 + 2)*n + b];
				numeric dw = dpsi_out->data[(a*4 + 3)*n + b];

				// gradient with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
				dgate->corner_block.data[0] += dx * x; dgate->corner_block.data[1] += dx * w;
				dgate->center_block.data[0] += dy * y; dgate->center_block.data[1] += dy * z;
				dgate->center_block.data[2] += dz * y; dgate->center_block.data[3] += dz * z;
				dgate->corner_block.data[2] += dw * x; dgate->corner_block.data[3] += dw * w;

				// gradient with respect to input vector 'psi'
				dpsi->data[(a*4    )*n + b] = gate->corner_block.data[0] * dx + gate->corner_block.data[2] * dw;
				dpsi->data[(a*4 + 1)*n + b] = gate->center_block.data[0] * dy + gate->center_block.data[2] * dz;
				dpsi->data[(a*4 + 2)*n + b] = gate->center_block.data[1] * dy + gate->center_block.data[3] * dz;
				dpsi->data[(a*4 + 3)*n + b] = gate->corner_block.data[1] * dx + gate->corner_block.data[3] * dw;
			}
		}
	}
	else
	{
		memset(dgate->center_block.data, 0, sizeof(dgate->center_block.data));
		memset(dgate->corner_block.data, 0, sizeof(dgate->corner_block.data));

		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					numeric dx = dpsi_out->data[(((a*2    )*n + b)*2    )*o + c];
					numeric dy = dpsi_out->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric dz = dpsi_out->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric dw = dpsi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// gradient of target function with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
					dgate->corner_block.data[0] += dx * x; dgate->corner_block.data[1] += dx * w;
					dgate->center_block.data[0] += dy * y; dgate->center_block.data[1] += dy * z;
					dgate->center_block.data[2] += dz * y; dgate->center_block.data[3] += dz * z;
					dgate->corner_block.data[2] += dw * x; dgate->corner_block.data[3] += dw * w;

					// gradient with respect to input vector 'psi'
					dpsi->data[(((a*2    )*n + b)*2    )*o + c] = gate->corner_block.data[0] * dx + gate->corner_block.data[2] * dw;
					dpsi->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->center_block.data[0] * dy + gate->center_block.data[2] * dz;
					dpsi->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->center_block.data[1] * dy + gate->center_block.data[3] * dz;
					dpsi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->corner_block.data[1] * dx + gate->corner_block.data[3] * dw;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate to qubits 'i' and 'j' of a statevector array.
///
void apply_matchgate_to_array(const struct matchgate* gate, const int i, const int j, const struct statevector_array* restrict psi, struct statevector_array* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(psi->nstates == psi_out->nstates);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { .center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			}
			,.corner_block.data =
			{ 
				gate -> corner_block.data[0],gate -> corner_block.data[1],
				gate -> corner_block.data[2],gate -> corner_block.data[3],
			}
		};
		// flip i <-> j
		apply_matchgate_to_array(&gate_perm, j, i, psi, psi_out);
	}
	else if (j == i + 1)
	{
		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = ((intqs)1 << (psi->nqubits - 1 - j)) * psi->nstates;
		#pragma omp for collapse(2) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				psi_out->data[(a*4    )*n + b] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
				psi_out->data[(a*4 + 1)*n + b] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
				psi_out->data[(a*4 + 2)*n + b] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
				psi_out->data[(a*4 + 3)*n + b] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
			}
		}
	}
	else
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = ((intqs)1 << (psi->nqubits - 1 - j)) * psi->nstates;
		#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					psi_out->data[(((a*2    )*n + b)*2    )*o + c] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
					psi_out->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Gate gradients corresponding to backward pass for applying a two-qubit gate to qubits 'i' and 'j' of a statevector array.
///
void apply_matchgate_backward_array(const struct matchgate* gate, const int i, const int j, const struct statevector_array* restrict psi,
	const struct statevector* restrict dpsi_out, struct matchgate* dgates)
{
	assert(psi->nqubits == dpsi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { .center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			}
			,.corner_block.data =
			{ 
				gate->corner_block.data[0], gate->corner_block.data[1],
				gate->corner_block.data[2], gate->corner_block.data[3],
			}
		};

		struct matchgate* dgates_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nstates * sizeof(struct matchgate));

		// flip i <-> j
		apply_matchgate_backward_array(&gate_perm, j, i, psi, dpsi_out, dgates_perm);

		// undo transposition of first and second qubit wire for gate gradient
		for (int k = 0; k < psi->nstates; k++)
		{
			// undo transposition of first and second qubit wire for gate gradient
			dgates[k].corner_block.data[0] = dgates_perm[k].corner_block.data[0]; dgates[k].corner_block.data[1] = dgates_perm[k].corner_block.data[1];
			dgates[k].center_block.data[0] = dgates_perm[k].center_block.data[3]; dgates[k].center_block.data[1] = dgates_perm[k].center_block.data[2];
			dgates[k].center_block.data[2] = dgates_perm[k].center_block.data[1]; dgates[k].center_block.data[3] = dgates_perm[k].center_block.data[0];
			dgates[k].corner_block.data[2] = dgates_perm[k].corner_block.data[2]; dgates[k].corner_block.data[3] = dgates_perm[k].corner_block.data[3];
		}

		aligned_free(dgates_perm);
	}
	else if (j == i + 1)
	{
		for (int k = 0; k < psi->nstates; k++) {
			memset(dgates[k].center_block.data, 0, sizeof(dgates[k].center_block.data));
			memset(dgates[k].corner_block.data, 0, sizeof(dgates[k].corner_block.data));
		}

		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(2) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric dx = dpsi_out->data[(a*4    )*n + b];
				numeric dy = dpsi_out->data[(a*4 + 1)*n + b];
				numeric dz = dpsi_out->data[(a*4 + 2)*n + b];
				numeric dw = dpsi_out->data[(a*4 + 3)*n + b];

				for (int k = 0; k < psi->nstates; k++)
				{
					numeric x = psi->data[((a*4    )*n + b)*psi->nstates + k];
					numeric y = psi->data[((a*4 + 1)*n + b)*psi->nstates + k];
					numeric z = psi->data[((a*4 + 2)*n + b)*psi->nstates + k];
					numeric w = psi->data[((a*4 + 3)*n + b)*psi->nstates + k];

					// gradient with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
					dgates[k].corner_block.data[0] += dx * x; dgates[k].corner_block.data[1] += dx * w;
					dgates[k].center_block.data[0] += dy * y; dgates[k].center_block.data[1] += dy * z;
					dgates[k].center_block.data[2] += dz * y; dgates[k].center_block.data[3] += dz * z;
					dgates[k].corner_block.data[2] += dw * x; dgates[k].corner_block.data[3] += dw * w;
				}
			}
		}
	}
	else
	{
		for (int k = 0; k < psi->nstates; k++) {
			memset(dgates[k].center_block.data, 0, sizeof(dgates[k].center_block.data));
			memset(dgates[k].corner_block.data, 0, sizeof(dgates[k].corner_block.data));
		}

		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric dx = dpsi_out->data[(((a*2    )*n + b)*2    )*o + c];
					numeric dy = dpsi_out->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric dz = dpsi_out->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric dw = dpsi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					for (int k = 0; k < psi->nstates; k++)
					{
						numeric x = psi->data[((((a*2    )*n + b)*2    )*o + c)*psi->nstates + k];
						numeric y = psi->data[((((a*2    )*n + b)*2 + 1)*o + c)*psi->nstates + k];
						numeric z = psi->data[((((a*2 + 1)*n + b)*2    )*o + c)*psi->nstates + k];
						numeric w = psi->data[((((a*2 + 1)*n + b)*2 + 1)*o + c)*psi->nstates + k];

						// gradient of target function with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
						dgates[k].corner_block.data[0] += dx * x; dgates[k].corner_block.data[1] += dx * w;
						dgates[k].center_block.data[0] += dy * y; dgates[k].center_block.data[1] += dy * z;
						dgates[k].center_block.data[2] += dz * y; dgates[k].center_block.data[3] += dz * z;
						dgates[k].corner_block.data[2] += dw * x; dgates[k].corner_block.data[3] += dw * w;
					}
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate "placeholder" acting on qubits 'i' and 'j' of a statevector.
/// 
/// Outputs a statevector array containing 8 vectors, corresponding to the placeholder matchgate entries.
///
void apply_matchgate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);
	assert(psi_out->nstates == 8);

	memset(psi_out->data, 0, ((size_t)1 << psi_out->nqubits) * psi_out->nstates * sizeof(numeric));

	if (i < j)
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// center block

					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 0] = y;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 1] = z;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 2] = y;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 3] = z;

					// corner block
					
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 0] = x;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 1] = w;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 2] = x;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 3] = w;
				}
			}
		}
	}
	else // i > j
	{
		const intqs m = (intqs)1 << j;
		const intqs n = (intqs)1 << (i - j - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - i);
		#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];


					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 0] = z;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 1] = y;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 2] = z;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 3] = y;

					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 0] = x;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 1] = w;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 2] = x;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 3] = w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate "placeholder" acting on qubits 'i' and 'j' of a statevector.
/// 
/// Outputs a statevector array containing 16 vectors, corresponding to the placeholder gate entries.
///
void apply_gate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);
	assert(psi_out->nstates == 16);

	memset(psi_out->data, 0, ((size_t)1 << psi_out->nqubits) * psi_out->nstates * sizeof(numeric));

	if (i < j)
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		#pragma omp for collapse(3)
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// equivalent to outer product with 4x4 identity matrix and transpositions
					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 0] = x;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 1] = y;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 2] = z;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 3] = w;
							
						}
					}
				}
			}
		}
	}
	else // i > j
	{
		const intqs m = (intqs)1 << j;
		const intqs n = (intqs)1 << (i - j - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - i);
		#pragma omp for collapse(3)
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// equivalent to outer product with 4x4 identity matrix and transpositions
					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 0] = x;
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 1] = z;
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 2] = y;  // note: flipping y <-> z
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 3] = w;
						}
					}
				}
			}
		}
	}
}

#else

//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit matchgate to qubits 'i' and 'j' of a statevector.
///
void apply_matchgate(const struct matchgate* gate, const int i, const int j, const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { .center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			}
			,.corner_block.data =
			{ 
				gate->corner_block.data[0],gate -> corner_block.data[1],
				gate->corner_block.data[2],gate -> corner_block.data[3],
			}
		};
		// flip i <-> j
		apply_matchgate(&gate_perm, j, i, psi, psi_out);
	}
	else if (j == i + 1)
	{
		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(2) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				psi_out->data[(a*4    )*n + b] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
				psi_out->data[(a*4 + 1)*n + b] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
				psi_out->data[(a*4 + 2)*n + b] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
				psi_out->data[(a*4 + 3)*n + b] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
			}
		}
	}
	else
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					psi_out->data[(((a*2    )*n + b)*2    )*o + c] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
					psi_out->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass for applying a two-qubit gate to qubits 'i' and 'j' of a statevector.
///
void apply_matchgate_backward(const struct matchgate* gate, const int i, const int j, const struct statevector* restrict psi,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate* dgate)
{
	assert(psi->nqubits == dpsi_out->nqubits);
	assert(psi->nqubits == dpsi->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { 
			.center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			},
			.corner_block.data =
			{ 
				gate->corner_block.data[0], gate->corner_block.data[1],
				gate->corner_block.data[2], gate->corner_block.data[3],
			}
		};

		struct matchgate dgate_perm;

		// flip i <-> j
		apply_matchgate_backward(&gate_perm, j, i, psi, dpsi_out, dpsi, &dgate_perm);

		// undo transposition of first and second qubit wire for gate gradient
		dgate->corner_block.data[0] = dgate_perm.corner_block.data[0]; dgate->corner_block.data[1] = dgate_perm.corner_block.data[1];
		dgate->center_block.data[0] = dgate_perm.center_block.data[3]; dgate->center_block.data[1] = dgate_perm.center_block.data[2];
		dgate->center_block.data[2] = dgate_perm.center_block.data[1]; dgate->center_block.data[3] = dgate_perm.center_block.data[0];
		dgate->corner_block.data[2] = dgate_perm.corner_block.data[2]; dgate->corner_block.data[3] = dgate_perm.corner_block.data[3];
	}
	else if (j == i + 1)
	{
		memset(dgate->center_block.data, 0, sizeof(dgate->center_block.data));
		memset(dgate->corner_block.data, 0, sizeof(dgate->corner_block.data));

		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(2) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				numeric dx = dpsi_out->data[(a*4    )*n + b];
				numeric dy = dpsi_out->data[(a*4 + 1)*n + b];
				numeric dz = dpsi_out->data[(a*4 + 2)*n + b];
				numeric dw = dpsi_out->data[(a*4 + 3)*n + b];

				// gradient with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
				dgate->corner_block.data[0] += dx * x; dgate->corner_block.data[1] += dx * w;
				dgate->center_block.data[0] += dy * y; dgate->center_block.data[1] += dy * z;
				dgate->center_block.data[2] += dz * y; dgate->center_block.data[3] += dz * z;
				dgate->corner_block.data[2] += dw * x; dgate->corner_block.data[3] += dw * w;

				// gradient with respect to input vector 'psi'
				dpsi->data[(a*4    )*n + b] = gate->corner_block.data[0] * dx + gate->corner_block.data[2] * dw;
				dpsi->data[(a*4 + 1)*n + b] = gate->center_block.data[0] * dy + gate->center_block.data[2] * dz;
				dpsi->data[(a*4 + 2)*n + b] = gate->center_block.data[1] * dy + gate->center_block.data[3] * dz;
				dpsi->data[(a*4 + 3)*n + b] = gate->corner_block.data[1] * dx + gate->corner_block.data[3] * dw;
			}
		}
	}
	else
	{
		memset(dgate->center_block.data, 0, sizeof(dgate->center_block.data));
		memset(dgate->corner_block.data, 0, sizeof(dgate->corner_block.data));

		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					numeric dx = dpsi_out->data[(((a*2    )*n + b)*2    )*o + c];
					numeric dy = dpsi_out->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric dz = dpsi_out->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric dw = dpsi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// gradient of target function with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
					dgate->corner_block.data[0] += dx * x; dgate->corner_block.data[1] += dx * w;
					dgate->center_block.data[0] += dy * y; dgate->center_block.data[1] += dy * z;
					dgate->center_block.data[2] += dz * y; dgate->center_block.data[3] += dz * z;
					dgate->corner_block.data[2] += dw * x; dgate->corner_block.data[3] += dw * w;

					// gradient with respect to input vector 'psi'
					dpsi->data[(((a*2    )*n + b)*2    )*o + c] = gate->corner_block.data[0] * dx + gate->corner_block.data[2] * dw;
					dpsi->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->center_block.data[0] * dy + gate->center_block.data[2] * dz;
					dpsi->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->center_block.data[1] * dy + gate->center_block.data[3] * dz;
					dpsi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->corner_block.data[1] * dx + gate->corner_block.data[3] * dw;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate to qubits 'i' and 'j' of a statevector array.
///
void apply_matchgate_to_array(const struct matchgate* gate, const int i, const int j, const struct statevector_array* restrict psi, struct statevector_array* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(psi->nstates == psi_out->nstates);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { .center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			}
			,.corner_block.data =
			{ 
				gate -> corner_block.data[0],gate -> corner_block.data[1],
				gate -> corner_block.data[2],gate -> corner_block.data[3],
			}
		};
		// flip i <-> j
		apply_matchgate_to_array(&gate_perm, j, i, psi, psi_out);
	}
	else if (j == i + 1)
	{
		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = ((intqs)1 << (psi->nqubits - 1 - j)) * psi->nstates;
		//#pragma omp for collapse(2) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				psi_out->data[(a*4    )*n + b] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
				psi_out->data[(a*4 + 1)*n + b] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
				psi_out->data[(a*4 + 2)*n + b] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
				psi_out->data[(a*4 + 3)*n + b] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
			}
		}
	}
	else
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = ((intqs)1 << (psi->nqubits - 1 - j)) * psi->nstates;
		//#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					psi_out->data[(((a*2    )*n + b)*2    )*o + c] = gate->corner_block.data[0] * x + gate->corner_block.data[1] * w;
					psi_out->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->center_block.data[0] * y + gate->center_block.data[1] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->center_block.data[2] * y + gate->center_block.data[3] * z;
					psi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->corner_block.data[2] * x + gate->corner_block.data[3] * w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Gate gradients corresponding to backward pass for applying a two-qubit gate to qubits 'i' and 'j' of a statevector array.
///
void apply_matchgate_backward_array(const struct matchgate* gate, const int i, const int j, const struct statevector_array* restrict psi,
	const struct statevector* restrict dpsi_out, struct matchgate* dgates)
{
	assert(psi->nqubits == dpsi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct matchgate gate_perm = { .center_block.data =
			{
				gate->center_block.data[3], gate->center_block.data[2], 
				gate->center_block.data[1], gate->center_block.data[0], 
			}
			,.corner_block.data =
			{ 
				gate->corner_block.data[0], gate->corner_block.data[1],
				gate->corner_block.data[2], gate->corner_block.data[3],
			}
		};

		struct matchgate* dgates_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nstates * sizeof(struct matchgate));

		// flip i <-> j
		apply_matchgate_backward_array(&gate_perm, j, i, psi, dpsi_out, dgates_perm);

		// undo transposition of first and second qubit wire for gate gradient
		for (int k = 0; k < psi->nstates; k++)
		{
			// undo transposition of first and second qubit wire for gate gradient
			dgates[k].corner_block.data[0] = dgates_perm[k].corner_block.data[0]; dgates[k].corner_block.data[1] = dgates_perm[k].corner_block.data[1];
			dgates[k].center_block.data[0] = dgates_perm[k].center_block.data[3]; dgates[k].center_block.data[1] = dgates_perm[k].center_block.data[2];
			dgates[k].center_block.data[2] = dgates_perm[k].center_block.data[1]; dgates[k].center_block.data[3] = dgates_perm[k].center_block.data[0];
			dgates[k].corner_block.data[2] = dgates_perm[k].corner_block.data[2]; dgates[k].corner_block.data[3] = dgates_perm[k].corner_block.data[3];
		}

		aligned_free(dgates_perm);
	}
	else if (j == i + 1)
	{
		for (int k = 0; k < psi->nstates; k++) {
			memset(dgates[k].center_block.data, 0, sizeof(dgates[k].center_block.data));
			memset(dgates[k].corner_block.data, 0, sizeof(dgates[k].corner_block.data));
		}

		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(2) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric dx = dpsi_out->data[(a*4    )*n + b];
				numeric dy = dpsi_out->data[(a*4 + 1)*n + b];
				numeric dz = dpsi_out->data[(a*4 + 2)*n + b];
				numeric dw = dpsi_out->data[(a*4 + 3)*n + b];

				for (int k = 0; k < psi->nstates; k++)
				{
					numeric x = psi->data[((a*4    )*n + b)*psi->nstates + k];
					numeric y = psi->data[((a*4 + 1)*n + b)*psi->nstates + k];
					numeric z = psi->data[((a*4 + 2)*n + b)*psi->nstates + k];
					numeric w = psi->data[((a*4 + 3)*n + b)*psi->nstates + k];

					// gradient with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
					dgates[k].corner_block.data[0] += dx * x; dgates[k].corner_block.data[1] += dx * w;
					dgates[k].center_block.data[0] += dy * y; dgates[k].center_block.data[1] += dy * z;
					dgates[k].center_block.data[2] += dz * y; dgates[k].center_block.data[3] += dz * z;
					dgates[k].corner_block.data[2] += dw * x; dgates[k].corner_block.data[3] += dw * w;
				}
			}
		}
	}
	else
	{
		for (int k = 0; k < psi->nstates; k++) {
			memset(dgates[k].center_block.data, 0, sizeof(dgates[k].center_block.data));
			memset(dgates[k].corner_block.data, 0, sizeof(dgates[k].corner_block.data));
		}

		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric dx = dpsi_out->data[(((a*2    )*n + b)*2    )*o + c];
					numeric dy = dpsi_out->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric dz = dpsi_out->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric dw = dpsi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					for (int k = 0; k < psi->nstates; k++)
					{
						numeric x = psi->data[((((a*2    )*n + b)*2    )*o + c)*psi->nstates + k];
						numeric y = psi->data[((((a*2    )*n + b)*2 + 1)*o + c)*psi->nstates + k];
						numeric z = psi->data[((((a*2 + 1)*n + b)*2    )*o + c)*psi->nstates + k];
						numeric w = psi->data[((((a*2 + 1)*n + b)*2 + 1)*o + c)*psi->nstates + k];

						// gradient of target function with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
						dgates[k].corner_block.data[0] += dx * x; dgates[k].corner_block.data[1] += dx * w;
						dgates[k].center_block.data[0] += dy * y; dgates[k].center_block.data[1] += dy * z;
						dgates[k].center_block.data[2] += dz * y; dgates[k].center_block.data[3] += dz * z;
						dgates[k].corner_block.data[2] += dw * x; dgates[k].corner_block.data[3] += dw * w;
					}
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate "placeholder" acting on qubits 'i' and 'j' of a statevector.
/// 
/// Outputs a statevector array containing 8 vectors, corresponding to the placeholder matchgate entries.
///
void apply_matchgate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);
	assert(psi_out->nstates == 8);

	memset(psi_out->data, 0, ((size_t)1 << psi_out->nqubits) * psi_out->nstates * sizeof(numeric));

	if (i < j)
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// center block

					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 0] = y;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 1] = z;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 2] = y;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 3] = z;

					// corner block
					
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 0] = x;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 1] = w;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 2] = x;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 3] = w;
				}
			}
		}
	}
	else // i > j
	{
		const intqs m = (intqs)1 << j;
		const intqs n = (intqs)1 << (i - j - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - i);
		//#pragma omp for collapse(3) 
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];


					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 0] = z;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 0)*o + c)*2 + 0)*4 + 1] = y;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 2] = z;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 1)*o + c)*2 + 0)*4 + 3] = y;

					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 0] = x;
					psi_out->data[(((((a*2 + 0)*n + b)*2 + 0)*o + c)*2 + 1)*4 + 1] = w;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 2] = x;
					psi_out->data[(((((a*2 + 1)*n + b)*2 + 1)*o + c)*2 + 1)*4 + 3] = w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate "placeholder" acting on qubits 'i' and 'j' of a statevector.
/// 
/// Outputs a statevector array containing 16 vectors, corresponding to the placeholder gate entries.
///
void apply_gate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);
	assert(psi_out->nstates == 16);

	memset(psi_out->data, 0, ((size_t)1 << psi_out->nqubits) * psi_out->nstates * sizeof(numeric));

	if (i < j)
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		//#pragma omp for collapse(3)
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// equivalent to outer product with 4x4 identity matrix and transpositions
					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 0] = x;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 1] = y;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 2] = z;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 3] = w;
							
						}
					}
				}
			}
		}
	}
	else // i > j
	{
		const intqs m = (intqs)1 << j;
		const intqs n = (intqs)1 << (i - j - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - i);
		//#pragma omp for collapse(3)
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// equivalent to outer product with 4x4 identity matrix and transpositions
					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 0] = x;
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 1] = z;
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 2] = y;  // note: flipping y <-> z
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 3] = w;
						}
					}
				}
			}
		}
	}
}

#endif
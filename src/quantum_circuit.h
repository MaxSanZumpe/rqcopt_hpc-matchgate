#pragma once 

#include "matchgate.h"
#include "statevector.h"

//________________________________________________________________________________________________________________________
///
/// \brief Temporary cache required by backward pass of a quantum circuit,
/// storing the sequence of intermediate statevectors.
///
struct quantum_circuit_cache
{
	struct statevector* psi_list;
	int nqubits;
	int ngates;
};

int allocate_quantum_circuit_cache(const int nqubits, const int ngates, struct quantum_circuit_cache* cache);

void free_quantum_circuit_cache(struct quantum_circuit_cache* cache);


int apply_quantum_matchgate_circuit(const struct matchgate gates[], const int ngates, const int wires[],
	const struct statevector* restrict psi, struct statevector* restrict psi_out);

int quantum_matchgate_circuit_forward(const struct matchgate gates[], const int ngates, const int wires[],
	const struct statevector* restrict psi, struct quantum_circuit_cache* cache, struct statevector* restrict psi_out);

int quantum_matchgate_circuit_backward(const struct matchgate gates[], const int ngates, const int wires[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dgates[]);


#pragma once

#include "matchgate.h"
#include "quantum_circuit.h"
#include "statevector.h"


int apply_matchgate_brickwall_unitary(const struct matchgate vlist[], const int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct statevector* restrict psi_out);

int matchgate_brickwall_unitary_forward(const struct matchgate vlist[], int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct quantum_circuit_cache* cache, struct statevector* restrict psi_out);

int matchgate_brickwall_unitary_backward(const struct matchgate vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dvlist[]);

int matchgate_brickwall_unitary_backward_hessian(const struct matchgate vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dvlist[], numeric* hess);

int matchgate_ti_brickwall_unitary_backward_hessian(const struct matchgate vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate dvlist[], numeric* hess);

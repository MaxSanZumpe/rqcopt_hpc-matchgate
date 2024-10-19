#pragma once

#include "matchgate.h"
#include "statevector.h"


void apply_matchgate(const struct matchgate* gate, const int i, const int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);

void apply_matchgate_backward(const struct matchgate* gate, const int i, const int j, const struct statevector* restrict psi,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct matchgate* dgate);


void apply_matchgate_to_array(const struct matchgate* gate, const int i, const int j, const struct statevector_array* restrict psi, struct statevector_array* restrict psi_out);

void apply_matchgate_backward_array(const struct matchgate* gate, const int i, const int j, const struct statevector_array* restrict psi,
	const struct statevector* restrict dpsi_out, struct matchgate* dgates);


void apply_matchgate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out);


void apply_gate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out);
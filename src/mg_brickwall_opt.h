#pragma once

#include "matchgate_brickwall.h"
#include "matchgate_target.h"
#include "trust_region.h"


void optimize_matchgate_brickwall_circuit_hmat(linear_func ufunc, void* udata,
	const struct matchgate vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct matchgate vlist_opt[]);


#ifdef MPI
void mpi_optimize_matchgate_brickwall_circuit_hmat(int rank, int start, int end, linear_func ufunc, void* udata,
	const struct matchgate vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct matchgate vlist_opt[]);
#endif
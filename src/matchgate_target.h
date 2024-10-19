#pragma once

#include "matchgate.h"
#include "quantum_circuit.h"
#include "statevector.h"

//add restricts back in


typedef int (*linear_func)(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out);


int matchgate_circuit_unitary_target(linear_func ufunc, void* udata, 
	const struct matchgate gates[], const int ngates, const int wires[], const int nqubits, numeric* fval);

int parallel_matchgate_circuit_unitary_target(linear_func ufunc, void* udata, 
	const struct matchgate gates[], const int ngates, const int wires[], const int nqubits, numeric* fval);

int matchgate_circuit_unitary_target_and_gradient(linear_func ufunc, void* udata,
	const struct matchgate gates[], const int ngates, const int wires[], const int nqubits,
	numeric* fval, struct matchgate dgates[]);


int matchgate_brickwall_unitary_target(linear_func ufunc, void* udata,
	const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[],
	numeric* fval);


int matchgate_brickwall_unitary_target_and_gradient(linear_func ufunc, void* udata,
	const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[],
	numeric* fval, struct matchgate dvlist[]);
	

int matchgate_brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata,
	const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[],
	numeric* fval, struct matchgate dvlist[], numeric* hess);
	

int matchgate_ti_brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata,
	const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[],
	numeric* fval, struct matchgate dvlist[], numeric* hess);
	

int matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix(linear_func ufunc, void* udata,
	const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[],
	numeric* fval, double* grad_vec, double* H);


int parallel_matchgate_brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata, 
	const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[], 
	numeric* fval, struct matchgate dvlist[], numeric* hess);


#ifdef MPI

int mpi_parallel_matchgate_circuit_unitary_target(linear_func ufunc, void* udata,
	const struct matchgate gates[], const int ngates, const int wires[], int start, int end,
	const int nqubits, numeric* fval);

int mpi_matchgate_brickwall_unitary_target(linear_func ufunc, void* udata,
	const struct matchgate vlist[], const int nlayers, const int nqubits, const int* perms[],
	int start, int end, numeric* fval);

int mpi_parallel_matchgate_brickwall_unitary_target_gradient_hessian(const int start, const int end, 
	linear_func ufunc, void* udata, const struct matchgate vlist[], const int nlayers, const int nqubits, 
	const int* perms[], numeric* fval, struct matchgate dvlist[], numeric* hess);

int mpi_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix(const struct matchgate vlist[], const int nlayers,
	struct matchgate* dvlist, numeric* hess, double* grad_vec, double* H);

int mpi_gradient_vector_hessian_matrix(int rank, int start, int end, linear_func ufunc, void* udata, const struct matchgate* vlist, const int nlayers, const int nqubits,
    const int* perms[], numeric* f_val, double* grad_vec, double* H);

int mpi_brickwall_f(int start, int end, linear_func ufunc, void* udata, const struct matchgate* vlist, const int nlayers, const int nqubits,
    const int* perms[], numeric* f_val);

#endif
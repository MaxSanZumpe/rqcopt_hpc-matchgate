#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "mg_brickwall_opt.h"


struct f_target_data
{
	linear_func ufunc;
	void* udata;
	const int** perms;
	int nlayers;
	int nqubits;
};


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of target function evaluation.
///
static double f(const double* x, void* fdata, numeric* f_complex)
{
	struct f_target_data* data = fdata;

	numeric f;
	if (matchgate_brickwall_unitary_target(data->ufunc, data->udata, (const struct matchgate*)x, data->nlayers, data->nqubits, data->perms, &f) < 0) {
		fprintf(stderr, "target function evaluation failed internally\n");
		return -1;
	}

	*f_complex = f;

	return creal(f);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of brickwall circuit target function and gradient evaluation.
///
// static double f_deriv(const double* restrict x, void* fdata, double* restrict grad)
// {
// 	struct f_target_data* data = fdata;

// 	numeric fval;
// 	if (matchgate_brickwall_unitary_target_and_projected_gradient(data->ufunc, data->udata, (const struct matchgate*)x, data->nlayers, data->nqubits, data->perms, &fval, grad) < 0) {
// 		fprintf(stderr, "target function and derivative evaluation failed internally\n");
// 		return -1;
// 	}

// 	return creal(fval);
//  }


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of brickwall circuit target function, gradient and Hessian evaluation.
///
static double f_deriv_hess(const double* restrict x, void* fdata, numeric* complex_f, double* restrict grad, double* restrict hess)
{
	struct f_target_data* data = fdata;
	numeric f;
	if (matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix(data->ufunc, data->udata, (const struct matchgate*)x, data->nlayers, data->nqubits, data->perms, &f, grad, hess) < 0) {
		fprintf(stderr, "target function and derivative evaluation failed internally\n");
		return -1;
	}

	*complex_f = f;

	return creal(f);
}


//________________________________________________________________________________________________________________________
///
/// \brief Retraction, with tangent direction represented as anti-symmetric matrices.
///
static void retract_matchgate_unitary_list(const double* restrict x, const double* restrict eta, void* rdata, double* restrict xs)
{
	const int nlayers = *((int*)rdata);
	assert(nlayers > 0);

	const struct matchgate* vlist  = (const struct matchgate*)x;
	struct matchgate* retractvlist = (struct matchgate*)xs;

	for (int i = 0; i < nlayers; i++)
	{
		retract_matchgate(&vlist[i], &eta[i * num_tangent_params], &retractvlist[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Optimize the quantum gates in a brickwall layout to approximate
/// a unitary matrix `U` using a trust-region method.
///
void optimize_matchgate_brickwall_circuit_hmat(linear_func ufunc, void* udata,
	const struct matchgate vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, numeric* f_citer, struct matchgate vlist_opt[])
{
	// target function data
	struct f_target_data fdata = {
		.ufunc   = ufunc,
		.udata   = udata,
		.perms   = perms,
		.nlayers = nlayers,
		.nqubits = nqubits,
	};

	// TODO: quantify error by spectral norm
	params->g_func = NULL;
	params->g_data = NULL;
	params->g_iter = NULL;

	// perform optimization
	int rdata = nlayers;
	
	riemannian_trust_region_optimize_hmat(f, f_deriv_hess, &fdata,retract_matchgate_unitary_list, &rdata,
		nlayers * 8, (const double*)vlist_start, nlayers * 8 * 2, params, niter, f_iter, f_citer, (double*)vlist_opt);
}


#ifdef MPI
#include <mpi.h>

struct mpi_f_target_data
{
	linear_func ufunc;
	void* udata;
	const int** perms;
	int nlayers;
	int nqubits;
	int rank;
	int start;
	int end;
};


double mpi_f(const double* restrict x, void* fdata)
{   
	numeric f;
    struct mpi_f_target_data* data = fdata;
    MPI_Bcast((void*)x, 2*8*data->nlayers, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    mpi_brickwall_f(data->start, data->end, data->ufunc, data->udata, (const struct matchgate*)x, data->nlayers, data->nqubits, data->perms, &f);
	
	return creal(f);
}

double mpi_f_deriv_hess(const double* x, void* fdata, double* restrict grad, double* restrict hess)
{   
	numeric f;
	struct mpi_f_target_data* data = fdata;
    MPI_Bcast((void*)x, 2*8*data->nlayers, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (mpi_gradient_vector_hessian_matrix(data->rank, data->start, data->end, data->ufunc, data->udata, (const struct matchgate*)x, data->nlayers, data->nqubits, data->perms, &f, grad, hess) < 0) {
		fprintf(stderr, " MPI target function and derivative evaluation failed internally\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	
	return creal(f);
}


void mpi_optimize_matchgate_brickwall_circuit_hmat(int rank, int start, int end, linear_func ufunc, void* udata,
	const struct matchgate vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct matchgate vlist_opt[])
{	
	// target function data
	struct mpi_f_target_data fdata = {
		.ufunc   = ufunc,
		.udata   = udata,
		.perms   = perms,
		.nlayers = nlayers,
		.nqubits = nqubits,
		.rank    = rank,
		.start   = start,
		.end     = end,
	};


	// TODO: quantify error by spectral norm
	params->g_func = NULL;
	params->g_data = NULL;
	params->g_iter = NULL;

	// perform optimization
	int rdata = nlayers;

	
	mpi_riemannian_trust_region_optimize_hmat(rank, mpi_f, mpi_f_deriv_hess, &fdata, retract_matchgate_unitary_list, &rdata,
		nlayers * 8, (const double*)vlist_start, nlayers * 8 * 2, params, niter, f_iter, (double*)vlist_opt);
}

#endif

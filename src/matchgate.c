#include <memory.h>
#include <stdio.h>
#include "matchgate.h"
#include "util.h"


void compare_matchgate_arrays(const struct matchgate* m, const struct matchgate* n, int d){

	printf("\n");
	int i, k;
	for (i = 0; i < d; i++){
		printf("\nMatchgate %d:\n", i);
		for (k = 0; k < 4; k++){
			printf("Center %d: %lf + %lfi  ", k, creal(m[i].center_block.data[k]), cimag(m[i].center_block.data[k]));
        	printf("           %lf + %lfi\n", creal(n[i].center_block.data[k]), cimag(n[i].center_block.data[k]));
		}

		for (k = 0; k < 4; k++){
			printf("Corner %d (%d): %lf + %lfi  ", k, k + 4, creal(m[i].corner_block.data[k]), cimag(m[i].corner_block.data[k]));
       		printf("           %lf + %lfi\n", creal(n[i].corner_block.data[k]), cimag(n[i].corner_block.data[k]));
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Set matrix (2x2) 'a' to the zero matrix.
///
void zero_mat2x2(struct mat2x2* a)
{
	memset(a->data, 0, sizeof(a->data));
}


//________________________________________________________________________________________________________________________
///
/// \brief Set matrix(2x2) 'a' to the identity matrix.
///
void identity_mat2x2(struct mat2x2* a)
{
	zero_mat2x2(a);

	a->data[ 0] = 1;
	a->data[ 3] = 1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Set matchgate 'a' to zero.
///
void zero_matchgate(struct matchgate* a)
{
	memset(a->center_block.data, 0, sizeof(a->center_block.data));
	memset(a->corner_block.data, 0, sizeof(a->center_block.data));
}


//________________________________________________________________________________________________________________________
///
/// \brief Set matchgate 'a' to the identity.
///
void identity_matchgate(struct matchgate* a)
{
	zero_matchgate(a);

	a->center_block.data[ 0] = 1;
	a->center_block.data[ 3] = 1;
	a->corner_block.data[ 0] = 1;
	a->corner_block.data[ 3] = 1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale matchgate 'a' by factor 'x'.
///
void scale_matchgate(struct matchgate* restrict a, const double x)
{
	for (int i = 0; i < 4; i++)
	{
		a->center_block.data[i] *= x;
		a->corner_block.data[i] *= x;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Complex-conjugate the entries of a matchgate.
///
void conjugate_matchgate(struct matchgate* a)
{
	#ifdef COMPLEX_CIRCUIT
	for (int i = 0; i < 4; i++)
	{
		a->center_block.data[i] = conj(a->center_block.data[i]);
		a->corner_block.data[i] = conj(a->corner_block.data[i]);
	}
	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Add matchgate 'b' to matchgate 'a'.
///
void add_matchgate(struct matchgate* restrict a, const struct matchgate* restrict b)
{
	for (int i = 0; i < 4; i++)
	{
		a->center_block.data[i] += b->center_block.data[i];
		a->corner_block.data[i] += b->corner_block.data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Subtract matchgate 'b' from matchgate 'a'.
///
void sub_matchgate(struct matchgate* restrict a, const struct matchgate* restrict b)
{
	for (int i = 0; i < 4; i++)
	{
		a->center_block.data[i] -= b->center_block.data[i];
		a->corner_block.data[i] -= b->corner_block.data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Add matchgates 'a' and 'b' and store the result in 'c'.
///
void add_matchgates(const struct matchgate* restrict a, const struct matchgate* restrict b, struct matchgate* restrict c)
{
	for (int i = 0; i < 4; i++)
	{
		c->center_block.data[i] = a->center_block.data[i] + b->center_block.data[i];
		c->corner_block.data[i] = a->corner_block.data[i] + b->corner_block.data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Subtract matchgates 'a' and 'b' and store the result in 'c'.
///
void sub_matchgates(const struct matchgate* restrict a, const struct matchgate* restrict b, struct matchgate* restrict c)
{
	for (int i = 0; i < 4; i++)
	{
		c->center_block.data[i] = a->center_block.data[i] - b->center_block.data[i];
		c->corner_block.data[i] = a->corner_block.data[i] - b->corner_block.data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Transpose a matchgate.
///
void transpose_matchgate(const struct matchgate* restrict a, struct matchgate* restrict at)
{
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			at->center_block.data[2*i + j] = a->center_block.data[2*j + i];
			at->corner_block.data[2*i + j] = a->corner_block.data[2*j + i];
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the adjoint (conjugate transpose) matchgate.
///
void adjoint_matchgate(const struct matchgate* restrict a, struct matchgate* restrict ah)
{
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			ah->center_block.data[2*i + j] = conj(a->center_block.data[2*j + i]);
			ah->corner_block.data[2*i + j] = conj(a->corner_block.data[2*j + i]);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Symmetrize a matchgate by projecting it onto the symmetric subspace.
///
void symm_matchgate(const struct matchgate* restrict w, struct matchgate* restrict z)
{
	// diagonal entries
    z->corner_block.data[ 0] = creal(w->corner_block.data[ 0]);
	z->center_block.data[ 0] = creal(w->center_block.data[ 0]);
    z->center_block.data[ 3] = creal(w->center_block.data[ 3]);
    z->corner_block.data[ 3] = creal(w->corner_block.data[ 3]);
	

	// upper triangular part
	z->corner_block.data[1] = 0.5 * (w->corner_block.data[1] + conj(w->corner_block.data[2]));
    z->center_block.data[1] = 0.5 * (w->center_block.data[1] + conj(w->center_block.data[2])); 

	// lower triangular part
	z->corner_block.data[2] = conj(z->corner_block.data[1]);
    z->center_block.data[2] = conj(z->center_block.data[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Antisymmetrize a matchgate by projecting it onto the antisymmetric (skew-symmetric) subspace.
///
void antisymm_matchgate(const struct matchgate* restrict w, struct matchgate* restrict z)
{
	// diagonal entries
	#ifdef COMPLEX_CIRCUIT
	z->corner_block.data[ 0] = I * cimag(w->corner_block.data[ 0]);
	z->center_block.data[ 0] = I * cimag(w->center_block.data[ 0]);
    z->center_block.data[ 3] = I * cimag(w->center_block.data[ 3]);
    z->corner_block.data[ 3] = I * cimag(w->corner_block.data[ 3]);
	#else
	z->corner_block.data[ 0] = 0;
	z->center_block.data[ 0] = 0;
    z->center_block.data[ 3] = 0;
    z->corner_block.data[ 3] = 0;
	#endif

	// upper triangular part
	z->corner_block.data[1] = 0.5 * (w->corner_block.data[1] - conj(w->corner_block.data[2]));
    z->center_block.data[1] = 0.5 * (w->center_block.data[1] - conj(w->center_block.data[2])); 

	// lower triangular part
	z->corner_block.data[2] = -conj(z->corner_block.data[1]);
    z->center_block.data[2] = -conj(z->center_block.data[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Isometrically map a real-valued square matrix or vector with 6 entries to an anti-symmetric matrix.
///
void real_to_antisymm_matchgate(const double* r, struct matchgate* w)
{
	#ifdef COMPLEX_CIRCUIT

	// diagonal entries
	w->center_block.data[0] = I * r[0];
	w->center_block.data[3] = I * r[3];
	w->corner_block.data[0] = I * r[4];
	w->corner_block.data[3] = I * r[7];

	// upper triangular part
	w->center_block.data[1] = 0.5*(r[1] - r[2]) + 0.5*I*(r[1] + r[2]);
	w->corner_block.data[1] = 0.5*(r[5] - r[6]) + 0.5*I*(r[5] + r[6]);

	// lower triangular part
	w->center_block.data[2] = -conj(w->center_block.data[1]);
	w->corner_block.data[2] = -conj(w->corner_block.data[1]);

	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Isometrically map an anti-symmetric or skew-symmetric matrix to a real-valued square matrix or vector with 6 entries.
///
void antisymm_matchgate_to_real(const struct matchgate* w, double* r)
{
	#ifdef COMPLEX_CIRCUIT

	for (int i = 0; i < 4; i++)
	{
		r[i    ] = creal(w->center_block.data[i]) + cimag(w->center_block.data[i]);
		r[i + 4] = creal(w->corner_block.data[i]) + cimag(w->corner_block.data[i]);
	}

	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Map a real-valued square matrix (unitary case) or a real vector with 6 entries (orthogonal case)
/// to a tangent vector of the unitary/orthogonal matrix manifold at point 'v'.
///
void real_to_tangent_matchgate(const double* r, const struct matchgate* restrict v, struct matchgate* restrict z)
{
	struct matchgate a;
	real_to_antisymm_matchgate(r, &a);
	multiply_matchgates(v, &a, z);
}


//________________________________________________________________________________________________________________________
///
/// \brief Project a matchgate into the tangent space of the Stiefel Manifold at 'v' and return
/// a real-valued square matrix (unitary case) or a real vector with 6 entries (orthogonal case).
///
void tangent_matchgate_to_real(const struct matchgate* restrict v, const struct matchgate* restrict z, double* r)
{
	struct matchgate w, t;
	adjoint_matchgate(v, &w);
	multiply_matchgates(&w, z, &t);
	antisymm_matchgate(&t, &w);
	antisymm_matchgate_to_real(&w, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Project 'z' onto the tangent plane at the unitary or orthogonal matrix 'u'.
///
void project_tangent_matchgate(const struct matchgate* restrict u, const struct matchgate* restrict z, struct matchgate* restrict p)
{
	// formula remains valid for 'u' an isometry (element of the Stiefel manifold)

	struct matchgate v, w;

	adjoint_matchgate(u, &v);
	multiply_matchgates(&v, z, &w);   // w = u^{\dagger} @ z
	symm_matchgate(&w, &v);           // v = symm(w)
	multiply_matchgates(u, &v, &w);   // w = u @ v
	sub_matchgates(z, &w, p);         // p = z - w
}



void multiply_mat2x2(const struct mat2x2* a, const struct mat2x2* b, struct mat2x2* restrict c)
{
	zero_mat2x2(c);

	// straightforward implementation; not performance critical, so not switching to BLAS yet
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				c->data[2*i + k] += a->data[2*i + j] * b->data[2*j + k];
			}
		}
	}
}


void multiply_matchgates(const struct matchgate* restrict a, const struct matchgate* restrict b, struct matchgate* restrict c)
{
	zero_matchgate(c);

    struct mat2x2* temp_center = malloc(sizeof(struct mat2x2));
    struct mat2x2* temp_corner = malloc(sizeof(struct mat2x2));

    multiply_mat2x2(&a->center_block, &b->center_block, temp_center);
    multiply_mat2x2(&a->corner_block, &b->corner_block, temp_corner);

    for (int i = 0; i < 4; i++)
    {
        c->center_block.data[i] = temp_center->data[i];
        c->corner_block.data[i] = temp_corner->data[i]; 
    }

    free(temp_center);
    free(temp_corner);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute (a @ b @ c + c @ b @ a) / 2.
///
void symmetric_triple_matchgate_product(const struct matchgate* restrict a, const struct matchgate* restrict b, const struct matchgate* restrict c, struct matchgate* restrict ret)
{
	struct matchgate u, v;

	multiply_matchgates(a, b, &u);
	multiply_matchgates(&u, c, ret);

	multiply_matchgates(c, b, &u);
	multiply_matchgates(&u, a, &v);

	add_matchgate(ret, &v);
	scale_matchgate(ret, 0.5);
}


//________________________________________________________________________________________________________________________
///
/// \brief Swap two rows of a matrix (2x2).
///
static inline void swap_rows_mat2x2(struct mat2x2* a, int i, int j)
{
	for (int k = 0; k < 2; k++)
	{
		numeric tmp      = a->data[2*i + k];
		a->data[2*i + k] = a->data[2*j + k];
		a->data[2*j + k] = tmp;
	}
}

//________________________________________________________________________________________________________________________
///
/// \brief Compute the inverse matrix (2x2) by Gaussian elimination, returning -1 if the matrix is singular.
///
int inverse_mat2x2(const struct mat2x2* restrict a, struct mat2x2* restrict ainv)
{
	// copy 'a' (for applying row operations)
	struct mat2x2 m;
	memcpy(m.data, a->data, sizeof(m.data));

	identity_mat2x2(ainv);

	for (int k = 0; k < 2; k++)
	{
		// search for pivot element in k-th column, starting from entry at (k, k)
		int i_max = k;
		double p = _abs(m.data[2*k + k]);
		for (int i = k + 1; i < 2; i++) {
			if (_abs(m.data[2*i + k]) > p) {
				i_max = i;
				p = _abs(m.data[2*i + k]);
			}
		}
		if (p == 0) {
			// matrix is singular
			return -1;
		}
		if (p < 1e-12) {
			fprintf(stderr, "warning: encountered an almost singular matrix in 'inverse_matrix', p = %g\n", p);
		}

		// swap pivot row with current row
		if (i_max != k)
		{
			swap_rows_mat2x2(&m,   k, i_max);
			swap_rows_mat2x2(ainv, k, i_max);
		}

		for (int i = 0; i < 2; i++)
		{
			if (i == k) {
				continue;
			}

			numeric s = m.data[2*i + k] / m.data[2*k + k];

			// subtract 's' times k-th row from i-th row
			for (int j = k + 1; j < 2; j++) {
				m.data[2*i + j] -= s * m.data[2*k + j];
			}
			m.data[2*i + k] = 0;

			// apply same row operation to 'ainv'
			for (int j = 0; j < 2; j++) {
				ainv->data[2*i + j] -= s * ainv->data[2*k + j];
			}
		}

		// divide k-th row by m[k, k]
		for (int j = k + 1; j < 2; j++) {
			m.data[2*k + j] /= m.data[2*k + k];
		}
		for (int j = 0; j < 2; j++) {
			ainv->data[2*k + j] /= m.data[2*k + k];
		}
		m.data[2*k + k] = 1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inverse matchgate
///
int inverse_matchgate(const struct matchgate* a, struct matchgate* restrict ainv)
{
	struct mat2x2* temp_center = malloc(sizeof(struct mat2x2));
    struct mat2x2* temp_corner = malloc(sizeof(struct mat2x2));

	inverse_mat2x2(&a->center_block, temp_center);
	inverse_mat2x2(&a->corner_block, temp_corner);

	for (int i = 0; i < 4; i++)
	{
		ainv->center_block.data[i] = temp_center->data[i];
		ainv->corner_block.data[i] = temp_corner->data[i];
	}

	free(temp_center);
	free(temp_corner);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the unitary polar factor 'u' in the polar decomposition 'a = u p' of a matrix,
/// assuming that 'a' is not singular.
///
/// Reference:
///     Nicholas J. Higham
///     Computing the polar decomposition - with applications
///     SIAM J. Sci. Stat. Comput. 7, 1160 - 1174 (1986)
///
void polar_matchgate_factor(const struct matchgate* restrict a, struct matchgate* restrict u)
{
	memcpy(u->center_block.data, a->center_block.data, sizeof(u->center_block.data));
	memcpy(u->corner_block.data, a->corner_block.data, sizeof(u->corner_block.data));

	for (int k = 0; k < 14; k++)
	{
		// w = u^{-\dagger}
		struct matchgate v, w;
		inverse_matchgate(u, &v);
		adjoint_matchgate(&v, &w);

		// u = (u + w)/2
		add_matchgate(u, &w);
		scale_matchgate(u, 0.5);

		if (k >= 4) {
			// early stopping
			if (uniform_distance(4, u->center_block.data, w.center_block.data) < 1e-14 &&
			    uniform_distance(4, u->corner_block.data, w.corner_block.data) < 1e-14) {
				break;
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Retract the tangent space vector 'eta' onto the orthogonal or unitary matrix manifold at 'u'.
///
void retract_matchgate(const struct matchgate* restrict u, const double* restrict eta, struct matchgate* restrict v)
{
	struct matchgate z;
	real_to_antisymm_matchgate(eta, &z);
	// add identity matrix
	z.corner_block.data[ 0]++;
	z.center_block.data[ 0]++;
	z.center_block.data[ 3]++;
	z.corner_block.data[ 3]++;

	struct matchgate w;
	multiply_matchgates(u, &z, &w);

	polar_matchgate_factor(&w, v);
}




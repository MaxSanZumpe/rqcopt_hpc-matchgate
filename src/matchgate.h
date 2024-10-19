#pragma once

#include "config.h"

struct mat2x2
{
	numeric data[4]; 
};

struct matchgate
{
	struct mat2x2 center_block;
	struct mat2x2 corner_block;
};

void compare_matchgate_arrays(const struct matchgate* m, const struct matchgate* n, int d);

void identity_mat2x2(struct mat2x2* a);
void identity_matchgate(struct matchgate* a);


void scale_matchgate(struct matchgate* restrict a, const double x);


void conjugate_matchgate(struct matchgate* a);


void add_matchgate(struct matchgate* restrict a, const struct matchgate* restrict b);
void sub_matchgate(struct matchgate* restrict a, const struct matchgate* restrict b);


void add_matchgates(const struct matchgate* restrict a, const struct matchgate* restrict b, struct matchgate* restrict c);
void sub_matchgates(const struct matchgate* restrict a, const struct matchgate* restrict b, struct matchgate* restrict c);


void transpose_matchgate(const struct matchgate* restrict a, struct matchgate* restrict at);
void adjoint_matchgate(const struct matchgate* restrict a, struct matchgate* restrict ah);


void symm_matchgate(const struct matchgate* restrict w, struct matchgate* restrict z);
void antisymm_matchgate(const struct matchgate* restrict w, struct matchgate* restrict z);


#ifdef COMPLEX_CIRCUIT
static const int num_tangent_params = 8;
#else
static const int num_tangent_params = 6;
#endif


void real_to_antisymm_matchgate(const double* r, struct matchgate* w);
void antisymm_matchgate_to_real(const struct matchgate* w, double* r);

void real_to_tangent_matchgate(const double* r, const struct matchgate* restrict v, struct matchgate* restrict z);
void tangent_matchgate_to_real(const struct matchgate* restrict v, const struct matchgate* restrict z, double* r);


void project_tangent_matchgate(const struct matchgate* restrict u, const struct matchgate* restrict z, struct matchgate* restrict p);

void multiply_mat2x2(const struct mat2x2* a, const struct mat2x2* b, struct mat2x2* restrict c);
void multiply_matchgates(const struct matchgate* restrict a, const struct matchgate* restrict b, struct matchgate* restrict c);


void symmetric_triple_matchgate_product(const struct matchgate* restrict a, const struct matchgate* restrict b, const struct matchgate* restrict c, struct matchgate* restrict ret);

int inverse_mat2x2(const struct mat2x2* restrict a, struct mat2x2* restrict ainv);
int inverse_matchgate(const struct matchgate* restrict a, struct matchgate* restrict ainv);


void polar_matchgate_factor(const struct matchgate* restrict a, struct matchgate* restrict u);


void retract_matchgate(const struct matchgate* restrict u, const double* restrict eta, struct matchgate* restrict v);



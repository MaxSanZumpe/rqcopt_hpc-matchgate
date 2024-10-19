#include "config.h"
#include "matchgate.h"
#include "util.h"

#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif

char* test_adjoint_matchgate()
{
    struct matchgate w;
    for (int i = 0; i < 4; i++)
    {   
        #ifdef COMPLEX_CIRCUIT
        w.center_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
        w.corner_block.data[i] = (((2 + i) * 9821) % 733) / 131.0 - 1 + ((((9 + i) * 3571) % 541) / 113.0 - 1) * I;

        #else
		w.center_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
        w.corner_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
		#endif
    }

    struct matchgate uref;
    adjoint_matchgate(&w, &uref);

    struct matchgate wh;
    transpose_matchgate(&w, &wh);
    conjugate_matchgate(&wh);

    if (uniform_distance(4, uref.center_block.data, wh.center_block.data) > 1e-14
      || uniform_distance(4, uref.corner_block.data, wh.corner_block.data) > 1e-14) {
		return "adjoint matchgate does not agree with reference";
	}

    return 0;
}

char* test_symm_matchgate()
{
	struct matchgate w;
	for (int i = 0; i < 4; i++)
	{
		#ifdef COMPLEX_CIRCUIT
		w.center_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
        w.corner_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;

		#else
		w.center_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
        w.corner_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
		#endif
	}

	struct matchgate uref;
	symm_matchgate(&w, &uref);

	// reference calculation
	// add adjoint matrix to 'w'
	struct matchgate wh;
	adjoint_matchgate(&w, &wh);
	add_matchgate(&w, &wh);
	scale_matchgate(&w, 0.5);

	// compare
	if (uniform_distance(4, uref.center_block.data, w.center_block.data) > 1e-14
      || uniform_distance(4, uref.corner_block.data, w.corner_block.data) > 1e-14) {
		return "symmetriurefed matchgate does not agree with reference";
	}

	return 0;
}


char* test_antisymm_macthagte()
{
	struct matchgate w;
	for (int i = 0; i < 4; i++)
	{
		#ifdef COMPLEX_CIRCUIT
		w.center_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
        w.corner_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;

		#else
		w.center_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
        w.corner_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
		#endif
	}


	struct matchgate uref;
	antisymm_matchgate(&w, &uref);

	// reference calculation
	// subtract adjoint matrix from 'w'
	struct matchgate wh;
	adjoint_matchgate(&w, &wh);
	sub_matchgate(&w, &wh);
	scale_matchgate(&w, 0.5);

	// compare
	if (uniform_distance(4, uref.center_block.data, w.center_block.data) > 1e-14
      || uniform_distance(4, uref.corner_block.data, w.corner_block.data) > 1e-14) {
		return "symmetriurefed matchgate does not agree with reference";
	}

	return 0;
}


char* test_real_to_antisymm_matchgate()
{
	double* r = aligned_alloc(MEM_DATA_ALIGN, num_tangent_params * sizeof(double));
	for (int i = 0; i < num_tangent_params; i++)
	{
		r[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
	}

	struct matchgate w;
	real_to_antisymm_matchgate(r, &w);

	// 'w' must indeed be anti-symmetric
	struct matchgate uref;
	antisymm_matchgate(&w, &uref);
	if (uniform_distance(4, uref.center_block.data, w.center_block.data) > 1e-14
      || uniform_distance(4, uref.corner_block.data, w.corner_block.data) > 1e-14) {
		return "matrix returned by 'real_to_antisymm' is not anti-symmetric";
	}

	double* s = aligned_alloc(MEM_DATA_ALIGN, num_tangent_params * sizeof(double));
	antisymm_matchgate_to_real(&w, s);
	// 's' must match 'r'
	double d = 0;
	for (int i = 0; i < num_tangent_params; i++)
	{
		d = fmax(d, fabs(s[i] - r[i]));
	}
	if (d > 1e-14) {
		return "converting from real vector to anti-symmetric matrix and back does not result in original vector";
	}

	aligned_free(s);
	aligned_free(r);

	return 0;
}


char* test_real_to_tangent_matchgate()
{
	hid_t file = H5Fopen("../test/data/test_real_to_tangent_matchgate" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_multiply_matchgate failed";
	}

	struct matchgate v;
	if (read_hdf5_dataset(file, "v/center", H5T_NATIVE_DOUBLE, v.center_block.data) < 0) {
		return "reading matchgate(center) entries from disk failed";
	}

	if (read_hdf5_dataset(file, "v/corner", H5T_NATIVE_DOUBLE, v.corner_block.data) < 0) {
		return "reading matchgate(corner) entries from disk failed";
	}

	struct matchgate t;
	if (read_hdf5_dataset(file, "t/center", H5T_NATIVE_DOUBLE, t.center_block.data) < 0) {
		return "reading matchgate(center) entries from disk failed";
	}

	if (read_hdf5_dataset(file, "t/corner", H5T_NATIVE_DOUBLE, t.corner_block.data) < 0) {
		return "reading matchgate(corner) entries from disk failed";
	}

	struct mat2x2 id;
	struct matchgate adj, uni;
	identity_mat2x2(&id);
	adjoint_matchgate(&v, &adj);
	multiply_matchgates(&adj, &v, &uni);
	
	if (uniform_distance(4, uni.center_block.data, id.data) > 1e-12
      || uniform_distance(4, uni.corner_block.data, id.data) > 1e-12) {
		return "not unitary";
	}

	double* r = aligned_alloc(MEM_DATA_ALIGN, num_tangent_params * sizeof(double));
	struct matchgate w, ats, tang;
	multiply_matchgates(&adj, &t, &w);
	antisymm_matchgate(&w, &ats);
	//antisymm_matchgate_to_real(&ats, r);
	tangent_matchgate_to_real(&v, &t, r);
	project_tangent_matchgate(&v, &t, &tang);


	struct matchgate ref_tang;
	real_to_tangent_matchgate(r, &v, &ref_tang);

	// compare
	if (uniform_distance(4, tang.center_block.data, ref_tang.center_block.data) > 1e-14
      || uniform_distance(4, tang.corner_block.data, ref_tang.corner_block.data) > 1e-14) {
		return "tangent matchgate does not agree with reference";
	}
	
	aligned_free(r);

	H5Fclose(file);

	return 0;

}


char* test_multiply_matchgates()
{
	hid_t file = H5Fopen("../test/data/test_multiply_matchgate" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_multiply_matchgate failed";
	}

	struct matchgate a;
	if (read_hdf5_dataset(file, "a/center", H5T_NATIVE_DOUBLE, a.center_block.data) < 0) {
		return "reading matchgate(center) entries from disk failed";
	}

	if (read_hdf5_dataset(file, "a/corner", H5T_NATIVE_DOUBLE, a.corner_block.data) < 0) {
		return "reading matchgate(corners) entries from disk failed";
	}

	struct matchgate b;
	if (read_hdf5_dataset(file, "b/center", H5T_NATIVE_DOUBLE, b.center_block.data) < 0) {
		return "reading matchgate(center) entries from disk failed";
	}

	if (read_hdf5_dataset(file, "b/corner", H5T_NATIVE_DOUBLE, b.corner_block.data) < 0) {
		return "reading matchgate(corners) entries from disk failed";
	}

	struct matchgate cref;
	if (read_hdf5_dataset(file, "c/center", H5T_NATIVE_DOUBLE, cref.center_block.data) < 0) {
		return "reading matchgate(center) entries from disk failed";
	}

	if (read_hdf5_dataset(file, "c/corner", H5T_NATIVE_DOUBLE, cref.corner_block.data) < 0) {
		return "reading matchgate(corners) entries from disk failed";
	}

	struct matchgate c;
	multiply_matchgates(&a, &b, &c);

	// compare
	if (uniform_distance(4, c.center_block.data, cref.center_block.data) > 1e-12
	    || uniform_distance(4, c.corner_block.data, cref.corner_block.data) > 1e-12) {
		return  "matchgate product does not agree with reference";
	}

	H5Fclose(file);

	return 0;

}


char* test_inverse_mat2x2()
{
	struct mat2x2 a;
	for (int i = 0; i < 4; i++)
	{
		#ifdef COMPLEX_CIRCUIT
		a.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
		#endif
	}

	struct mat2x2 ainv;
	inverse_mat2x2(&a, &ainv);

	struct mat2x2 prod;
	multiply_mat2x2(&a, &ainv, &prod);

	struct mat2x2 id;
	identity_mat2x2(&id);

	// must be (close to) identity matrix
	if (uniform_distance(4, prod.data, id.data) > 1e-14) {
		return "matrix times its inverse is not close to identity";
	}

	// generate a singular matrix
	for (int j = 0; j < 2; j++)
	{
		a.data[2 + j] = a.data[j];
	}
	int ret = inverse_mat2x2(&a, &ainv);
	if (ret != -1) {
		return "missing singular matrix indicator";
	}

	return 0;
}


char* test_inverse_matchgate()
{
	struct matchgate a;
	for (int i = 0; i < 4; i++)
	{
		#ifdef COMPLEX_CIRCUIT
		a.center_block.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
		a.corner_block.data[i] = (((2 + i) * 5254) % 753) / 209.0- 1 + ((((9 + i) * 9024) % 541) / 500.0 - 1) * I;
		#endif
	}

	struct matchgate ainv;
	inverse_matchgate(&a, &ainv);

	struct matchgate prod;
	multiply_matchgates(&a, &ainv, &prod);

	struct matchgate id;
	identity_matchgate(&id);

	// must be (close to) identity matrix
	if (uniform_distance(4, prod.center_block.data, id.center_block.data) > 1e-12
	    || uniform_distance(4, prod.corner_block.data, id.corner_block.data) > 1e-12) {
	 	return  "matchgate inverse does not produce identity";
	}

	return 0;
}


char* test_project_tangent_matchgate()
{
	hid_t file = H5Fopen("../test/data/test_project_tangent_matchgate" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_project_tangent_matchgate failed";
	}

	struct matchgate u;
	if (read_hdf5_dataset(file, "u/center", H5T_NATIVE_DOUBLE, u.center_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}
	if (read_hdf5_dataset(file, "u/corner", H5T_NATIVE_DOUBLE, u.corner_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct matchgate uref;
	if (read_hdf5_dataset(file, "z/center", H5T_NATIVE_DOUBLE, uref.center_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}
	if (read_hdf5_dataset(file, "z/corner", H5T_NATIVE_DOUBLE, uref.corner_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct matchgate pref;
	if (read_hdf5_dataset(file, "p/center", H5T_NATIVE_DOUBLE, pref.center_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}
	if (read_hdf5_dataset(file, "p/corner", H5T_NATIVE_DOUBLE, pref.corner_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct matchgate p;
	project_tangent_matchgate(&u, &uref, &p);
	// compare
	if (uniform_distance(8, (numeric*)&p, (numeric*)&pref) > 1e-12) {
		return "projected matchgate does not agree with reference";
	}

	H5Fclose(file);

	return 0;
}

char* test_polar_matchgate_factor()
{
	hid_t file = H5Fopen("../test/data/test_polar_matchgate_factor" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_polar_factor failed";
	}

	struct matchgate a;
	if (read_hdf5_dataset(file, "a/center", H5T_NATIVE_DOUBLE, a.center_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}
	if (read_hdf5_dataset(file, "a/corner", H5T_NATIVE_DOUBLE, a.corner_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct matchgate uref;
	if (read_hdf5_dataset(file, "u/center", H5T_NATIVE_DOUBLE, uref.center_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}
	if (read_hdf5_dataset(file, "u/corner", H5T_NATIVE_DOUBLE, uref.corner_block.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct matchgate u;
	polar_matchgate_factor(&a, &u);

	// compare
	if (uniform_distance(8, (numeric*)&u, (numeric*)&uref) > 1e-12) {
		return "polar factor does not agree with reference";
	}

	H5Fclose(file);

	return 0;
}

#include <stdio.h>


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_adjoint_matchgate();
char* test_symm_matchgate();
char* test_antisymm_macthagte();
char* test_multiply_matchgates();
char* test_real_to_antisymm_matchgate();
char* test_real_to_tangent_matchgate();
char* test_inverse_mat2x2();
char* test_inverse_matchgate();
char* test_project_tangent_matchgate();
char* test_polar_matchgate_factor();

char* test_apply_matchgate();
char* test_apply_matchgate_backward();
char* test_apply_matchgate_backward_array();
char* test_apply_matchgate_to_array();
char* test_apply_matchgate_placeholder();

char* test_apply_matchgate_brickwall_unitary();
char* test_matchgate_brickwall_unitary_backward();
char* test_matchgate_brickwall_unitary_backward_hessian();

char* test_matchgate_circuit_unitary_target();
char* test_matchgate_brickwall_unitary_target();
char* test_matchgate_brickwall_unitary_target_gradient_hessian();
char* test_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix();
char* test_matchgate_ti_brickwall_unitary_target_gradient_hessian();
char* test_matchgate_ti_brickwall_unitary_target_gradient_hessian2();
char* test_matchgate_ti_brickwall_unitary_target_gradient_hessian3();
char* test_parallel_matchgate_brickwall_unitary_target_gradient_hessian();


#define TEST_FUNCTION_ENTRY(fname) { .func = fname, .name = #fname }

int main(){
    struct test tests[] = {
        TEST_FUNCTION_ENTRY(test_symm_matchgate),
		TEST_FUNCTION_ENTRY(test_adjoint_matchgate),
		TEST_FUNCTION_ENTRY(test_antisymm_macthagte),
		TEST_FUNCTION_ENTRY(test_multiply_matchgates),	
		TEST_FUNCTION_ENTRY(test_real_to_antisymm_matchgate),
		TEST_FUNCTION_ENTRY(test_real_to_tangent_matchgate),
		TEST_FUNCTION_ENTRY(test_inverse_mat2x2),
		TEST_FUNCTION_ENTRY(test_inverse_matchgate),
		TEST_FUNCTION_ENTRY(test_project_tangent_matchgate),
		TEST_FUNCTION_ENTRY(test_polar_matchgate_factor),
		TEST_FUNCTION_ENTRY(test_apply_matchgate),
		TEST_FUNCTION_ENTRY(test_apply_matchgate_to_array),
		TEST_FUNCTION_ENTRY(test_apply_matchgate_backward),
		TEST_FUNCTION_ENTRY(test_apply_matchgate_backward_array),
		TEST_FUNCTION_ENTRY(test_apply_matchgate_placeholder),
		TEST_FUNCTION_ENTRY(test_apply_matchgate_brickwall_unitary),
		TEST_FUNCTION_ENTRY(test_matchgate_brickwall_unitary_backward),
		TEST_FUNCTION_ENTRY(test_matchgate_brickwall_unitary_backward_hessian),
		TEST_FUNCTION_ENTRY(test_matchgate_circuit_unitary_target),
		TEST_FUNCTION_ENTRY(test_matchgate_brickwall_unitary_target),
		TEST_FUNCTION_ENTRY(test_matchgate_brickwall_unitary_target_gradient_hessian),
		TEST_FUNCTION_ENTRY(test_matchgate_brickwall_unitary_target_gradient_vector_hessian_matrix),
		TEST_FUNCTION_ENTRY(test_matchgate_ti_brickwall_unitary_target_gradient_hessian),
		//TEST_FUNCTION_ENTRY(test_matchgate_ti_brickwall_unitary_target_gradient_hessian2),
		TEST_FUNCTION_ENTRY(test_matchgate_ti_brickwall_unitary_target_gradient_hessian3),
		TEST_FUNCTION_ENTRY(test_parallel_matchgate_brickwall_unitary_target_gradient_hessian),
    };
    int num_tests = sizeof(tests) / sizeof(struct test);

	int num_pass = 0;
	for (int i = 0; i < num_tests; i++)
	{
		printf(".");
		char* msg = tests[i].func();
		if (msg == 0) {
			num_pass++;
		}
		else {
			printf("\nTest '%s' failed: %s\n", tests[i].name, msg);
		}
	}
	printf("\nNumber of successful tests: %i / %i\n", num_pass, num_tests);

	if (num_pass < num_tests)
	{
		printf("At least one test failed!\n");
	}
	else
	{
		printf("All tests passed.\n");
	}

	return num_pass != num_tests;
}

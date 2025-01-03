#include <stdio.h>
#include <omp.h>
#include <unistd.h> // For usleep

int main() {
    
    N = 10;
    int nqubits = 10;
    const long int m = 1 << nqubits;
    int shared_array[N];

    for (int i = 0; i < N; i++) {
        shared_array[i] = 0;
    }

    #pragma omp parallel for shared(shared_array)
    for (int i = 0; i < N; i++) {

        // Simulate some work
        usleep(500000); // Sleep for 500 milliseconds

        // Update shared resource
        shared_array[i] = i * i;

        printf("Thread %d finished index %d\n", thread_num, i);
    }

    printf("Parallel computation complete. Results:\n");
    for (int i = 0; i < N; i++) {
        printf("shared_array[%d] = %d\n", i, shared_array[i]);
    }

    return 0;
}

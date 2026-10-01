#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Compute vector sum C = A + B
void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{
    for (int i = 0; i < n; i++)
        h_C[i] = h_A[i] + h_B[i];
}

int main()
{
    int N = 1000000;

    // Memory allocation for h_A, h_B, and h_C
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    // Initialize random seed
    srand(time(NULL));

    // Fill h_A and h_B with random numbers
    for (int i = 0; i < N; i++)
    {
        h_A[i] = (float)rand() / RAND_MAX;  // Random float between 0 and 1
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Perform vector addition
    vecAdd(h_A, h_B, h_C, N);

    // Display the first few elements of h_A, h_B, and h_C
    printf("h_A, h_B, h_C :\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.3f %.3f %.3f", h_A[i], h_B[i], h_C[i]);
	printf("\n");
    }
    printf("\n");

    // Free allocated memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

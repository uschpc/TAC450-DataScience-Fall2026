#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA kernel for vector addition
__global__ void vecAddKernel(float *d_A, float *d_B, float *d_C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        d_C[i] = d_A[i] + d_B[i];
    }
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{

    int size = n* sizeof(float);
    // Allocate device memory for d_A, d_B, and d_C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
}

int main()
{
    int n = 1000000;
    
     // Memory allocation for h_A, h_B, and h_C
    float *h_A = (float *)malloc(n * sizeof(float));
    float *h_B = (float *)malloc(n * sizeof(float));
    float *h_C = (float *)malloc(n * sizeof(float));

    // Initialize random seed
    srand(time(NULL));  // Include <time.h> for time function

    // Fill h_A and h_B with random numbers
    for (int i = 0; i < n; i++)
    {
        h_A[i] = (float)rand() / RAND_MAX;  // Random float between 0 and 1
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Perform vector addition
    vecAdd(h_A, h_B, h_C, n);
    
    // Display the first few elements of h_A, h_B, and h_C
    printf("h_A, h_B, h_C :\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.3f %.3f %.3f", h_A[i], h_B[i], h_C[i]);
	    printf("\n");
    }
    printf("\n");

    // Free host and device memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

#include <stdio.h>

#define N 8
#define RADIUS 1

// Constant memory for stencil weights
__constant__ int d_weights[2 * RADIUS + 1];

// Kernel using constant memory
__global__ void stencil1D(int *in, int *out) {
    int i = threadIdx.x;

    int result = 0;
    for (int j = -RADIUS; j <= RADIUS; j++) {
        int idx = i + j;
        // Handle boundary
        if (idx >= 0 && idx < N) {
            result += d_weights[j + RADIUS] * in[idx];
        }
    }

    out[i] = result;
}

int main() {
    int h_in[N]  = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_out[N];

    int *d_in, *d_out;

    // Define stencil weights (e.g., [1, 1, 1])
    int h_weights[2 * RADIUS + 1] = {1, 1, 1};

    // Copy weights to constant memory
    cudaMemcpyToSymbol(d_weights, h_weights, sizeof(h_weights));

    // Allocate device memory
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    stencil1D<<<1, N>>>(d_in, d_out);

    // Copy result back
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Input : ");
    for (int i = 0; i < N; i++) printf("%d ", h_in[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) printf("%d ", h_out[i]);
    printf("\n");

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;  }

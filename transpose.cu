#include <stdio.h>
#define N 3  // 3x3 matrix

__global__ void transposer(int *input, int *output, int width) {
    __shared__ int tile[N][N];  // Fixed shared memory for 3x3

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * width) {
        int row = idx / width;
        int col = idx % width;

        // Load into shared memory (transpose here)
        tile[col][row] = input[row * width + col];

        __syncthreads();  // Synchronize to make sure all writes are complete

        // Now write from shared memory to output
        output[idx] = tile[row][col];
    }
}
int main() {
    int size = N * N * sizeof(int);
    int h_input[N * N], h_output[N * N];

    // Generate random input matrix
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_input[i] = rand() % 100;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    transposer<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Display original matrix
    printf("Original Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%2d ", h_input[i * N + j]);
        }
        printf("\n");
    }

    // Display transposed matrix
    printf("\nTransposed Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%2d ", h_output[i * N + j]);
        }
        printf("\n");    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

double timeStamp() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

__global__ void oddEvenKernel(int *arr, int n, int phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - 1) return;
    if ((idx & 1) == phase) {
        int a = arr[idx];
        int b = arr[idx + 1];
        if (a > b) {
            arr[idx] = b;
            arr[idx + 1] = a;
        }
    }
}

int cmp_int(const void *a, const void *b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

int main(int argc, char **argv) {
    int n = 8;
    if (argc > 1) n = atoi(argv[1]);
    if (n < 2) n = 2;

    int *h_in  = (int*)malloc(n * sizeof(int));
    int *h_cpu = (int*)malloc(n * sizeof(int));
    int *h_gpu = (int*)malloc(n * sizeof(int));

    // Initialize data: example small array if n==8 else random
    if (n == 8) {
        int sample[8] = {2, 8, 3, 7, 5, 9, 4, 6};
        for (int i = 0; i < n; ++i) h_in[i] = sample[i];
    } else {
        srand(123);
        for (int i = 0; i < n; ++i) h_in[i] = rand() % 100000;
    }

    for (int i = 0; i < n; ++i) {
        h_cpu[i] = h_in[i];
        h_gpu[i] = h_in[i];
    }

    int *d_arr = NULL;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_gpu, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int phase = 0; phase < n; ++phase) {
        int p = phase & 1; // 0 = even phase, 1 = odd phase
        oddEvenKernel<<<blocks, threads>>>(d_arr, n, p);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_gpu, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    double cpu_start = timeStamp();
    qsort(h_cpu, n, sizeof(int), cmp_int);
    double cpu_end = timeStamp();

    // Validate
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (h_cpu[i] != h_gpu[i]) { ok = 0; break; }
    }

    // Print results (for small n print arrays)
    if (n <= 64) {
        printf("Input : ");
        for (int i = 0; i < n; ++i) printf("%d ", h_in[i]);
        printf("\n");
        printf("GPU   : ");
        for (int i = 0; i < n; ++i) printf("%d ", h_gpu[i]);
        printf("\n");
        printf("CPU   : ");
        for (int i = 0; i < n; ++i) printf("%d ", h_cpu[i]);
        printf("\n");
    }

    printf("Validation: %s\n", ok ? "SUCCESS (GPU matches CPU)" : "FAIL (mismatch)");
    printf("Execution time (GPU kernels total): %.6f s\n", gpu_ms / 1000.0f);
    printf("Execution time (CPU qsort): %.6f s\n", cpu_end - cpu_start);

    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_in);
    free(h_cpu);
    free(h_gpu);

    return ok ? 0 : 1;
}

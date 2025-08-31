#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define THREADS 256

double timeStamp() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

_device_ int partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
    }
    int t = arr[i + 1]; arr[i + 1] = arr[high]; arr[high] = t;
    return i + 1;
}

_global_ void quicksortKernel(int *arr, int *lowStack, int *highStack, int *stackSize) {
    if (threadIdx.x == 0) {
        while (*stackSize > 0) {
            int high = highStack[*stackSize - 1];
            int low = lowStack[*stackSize - 1];
            atomicSub(stackSize, 1);
            if (low < high) {
                int pi = partition(arr, low, high);
                int pos = atomicAdd(stackSize, 1);
                lowStack[pos] = pi + 1; highStack[pos] = high;
                pos = atomicAdd(stackSize, 1);
                lowStack[pos] = low; highStack[pos] = pi - 1;
            }
        }
    }
}

int cmp_int(const void *a, const void *b) {
    int ia = (const int)a, ib = (const int)b;
    return (ia > ib) - (ia < ib);
}

int main() {
    int n = 1 << 10;
    int h_in = (int)malloc(n * sizeof(int));
    int h_gpu = (int)malloc(n * sizeof(int));
    int h_cpu = (int)malloc(n * sizeof(int));
    srand(123);
    for (int i = 0; i < n; i++) { h_in[i] = rand() % 10000; h_gpu[i] = h_in[i]; h_cpu[i] = h_in[i]; }

    int *d_arr, *d_lowStack, *d_highStack, *d_stackSize;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_lowStack, n * sizeof(int));
    cudaMalloc(&d_highStack, n * sizeof(int));
    cudaMalloc(&d_stackSize, sizeof(int));
    cudaMemcpy(d_arr, h_gpu, n * sizeof(int), cudaMemcpyHostToDevice);

    int h_lowStack[n], h_highStack[n], h_stackSize = 1;
    h_lowStack[0] = 0; h_highStack[0] = n - 1;
    cudaMemcpy(d_lowStack, h_lowStack, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_highStack, h_highStack, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop; float gpu_ms;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    quicksortKernel<<<1, THREADS>>>(d_arr, d_lowStack, d_highStack, d_stackSize);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_gpu, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    double cpu_start = timeStamp(); qsort(h_cpu, n, sizeof(int), cmp_int); double cpu_end = timeStamp();

    printf("Execution time (GPU): %.6f s\n", gpu_ms / 1000.0f);
    printf("Execution time (CPU): %.6f s\n", cpu_end - cpu_start);

    cudaFree(d_arr); cudaFree(d_lowStack); cudaFree(d_highStack); cudaFree(d_stackSize);
    free(h_in); free(h_gpu); free(h_cpu);
    return 0;
}
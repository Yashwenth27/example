#include<stdio.h>
#include<sys/time.h>

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void dotProduct(int *a,int *b,int *result,int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = tid;
    if (index<n){
        a[index]=a[index]*b[index];
    }
    __syncthreads();
    for(int stride=1;stride<n;stride*=2){
        if (index%(2*stride)==0 && (index+stride)<n){
            a[index]+=a[index+stride];
        }
    }
    __syncthreads();
    if(index==0){
        *result=a[0];
    }
}

int main(){
    int N = 5;

    int h_a[] = {10,20,30,40,50};
    int h_b[] = {1,2,3,4,5};
    int h_result = 0;

    int *d_a,*d_b,*d_result;

    cudaMalloc((void**)&d_a,N*sizeof(int));
    cudaMalloc((void**)&d_b,N*sizeof(int));
    cudaMalloc((void**)&d_result,sizeof(int));

    cudaMemcpy(d_a,h_a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,N*sizeof(int),cudaMemcpyHostToDevice);

    double gpuStart = cpuSecond();

    int threadsPerBlock = 5;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    dotProduct <<< blocks,threadsPerBlock >>> (d_a,d_b,d_result,N);

    cudaDeviceSynchronize();

    double gpuStop = cpuSecond();

    cudaMemcpy(&h_result,d_result,sizeof(int),cudaMemcpyDeviceToHost);

    printf("Dot Product: %d\n",h_result);
    printf("GPU Seconds: %f\n",gpuStop-gpuStart);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

}
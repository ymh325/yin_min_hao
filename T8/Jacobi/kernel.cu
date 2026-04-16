#include "Jacobi.h"

__global__ void jacobi_kernel(float *dA, float *dAnew, int n,int m){
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(i < n - 1 && j < m - 1){ 
        dAnew[i * m + j] = 0.25f * (dA[(i-1) * m + j] + dA[(i+1) * m + j] + 
                                    dA[i * m + (j-1)] + dA[i * m + (j+1)]);
    }
}

void Initial(float *&dA, float *&dAnew,int n,int m){
    cudaMalloc(&dA, n * m * sizeof(float));
    cudaMalloc(&dAnew, n * m * sizeof(float));
    cudaMemset(dAnew, 0, n * m * sizeof(float));
    float *tmp = (float*)malloc(n * m * sizeof(float));
    for(int i = 0; i <n; i++){
        for(int j = 0; j <m; j++){            
            if(0<i && i<n-1 && 0<j && j<m-1)tmp[i*m+j] = 1.0f;
            else tmp[i*m+j] = 0.0f;
        }
    }
    cudaMemcpy(dA, tmp, n * m * sizeof(float), cudaMemcpyHostToDevice);
    free(tmp);
}
void Lanuch_kernel(float *dA, float *dAnew, int n,int m){
    dim3 block(16, 16);
    dim3 grid((n-2 + block.x - 1) / block.x, (m-2 + block.y - 1) / block.y);
    jacobi_kernel<<<grid, block>>>(dA, dAnew, n, m);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
void Free(float *dA, float *dAnew){
    cudaFree(dA);
    cudaFree(dAnew);
}
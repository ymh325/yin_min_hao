#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

/*
Observations:

If we set aside the purpose of learning the concept of register tiling, 
the code of this V3 example version has certain flaws.

The main issue is that in the loop for copying data from global memory to shared memory, 
each thread reads the range of 0~Share_K, which actually causes redundant memory reads. 
Furthermore, only thread.x is utilized in the code.

This renders the data transfer from global memory to shared memory redundant,
as the effect of distributing the memory read workload across multiple threads is not achieved at all.


*/

/*--------------------------------------------------
    name: kernel_mul
    input:
        dA: input matrix A, size M * K
        dB: input matrix B, size K * N
        dC: output matrix C, size M * N
        M: number of rows of matrix A
        K: number of columns of matrix A and number of rows of matrix B
        N: number of columns of matrix B
    Function :
        Matrix multiplication kernel with shared memory
    return: None
----------------------------------------------------*/
template <int Share_M, int Share_N, int Share_K, int TM, int TN>
__global__ void kernel_mul_v3(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[Share_M * Share_K];
    __shared__ float SB[Share_K * Share_N];
    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    int width = (K + Share_K - 1) / Share_K;
    float tmp[TM * TN] = {0.0f};

    for (int ph = 0; ph < width; ph++)
    {

        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_k = 0; index_k < Share_K; index_k++)
            {
                if (indA + index_q < M && index_k + ph * Share_K < K)
                {
                    SA[(threadIdx.x * TM + index_q) * Share_K + index_k] = dA[(indA + index_q) * K + index_k + ph * Share_K];
                }
                else
                {
                    SA[(threadIdx.x * TM + index_q) * Share_K + index_k] = 0.0f;
                }
            }
        }
        __syncthreads();
        for (int index_v = 0; index_v < TN; index_v++)
        {
            for (int index_k = 0; index_k < Share_K; index_k++)
            {

                if (indB + index_v < N && index_k + ph * Share_K < K)
                {

                    SB[index_k * Share_N + threadIdx.y * TN + index_v] = dB[(index_k + ph * Share_K) * N + indB + index_v];
                }
                else
                {
                    SB[index_k * Share_N + threadIdx.y * TN + index_v] = 0.0f;
                }
            }
        }

        __syncthreads();
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                for (int index_k = 0; index_k < Share_K; index_k++)
                {
                    tmp[index_q * TN + index_v] += SA[(threadIdx.x * TM + index_q) * Share_K + index_k] * SB[index_k * Share_N + threadIdx.y * TN + index_v];
                }
            }
        }
        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + index_q) * N + indB + index_v] = tmp[index_q * TN + index_v];
            }
        }
    }
}
void cpu_matrix_mul(float *a, float *b, float *c, int M, int K, int N) {
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float sum = 0;
            for(int k=0;k<K;k++){
                sum += a[i*K + k] * b[k*N + j];
            }
            c[i*N + j] = sum;
        }
    }
}
bool verify_result(float *cpu_res, float *gpu_res, int size, float eps=1e-5) {
    for(int i=0; i<size; i++) {
        if(fabs(cpu_res[i] - gpu_res[i]) > eps) {
            printf("The %dth is not equal:CPU=%.4f, GPU=%.4f\n", i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    printf("\n Verification passed!\n");
    return true;
}
int main(){
    const int M = 1024, K = 1024, N = 1024; // Matrix size  
    const int BLOCK_X = 32, BLOCK_Y = 32 ;  // Block size
    const int TM = 4, TN = 4;               // Thread computation size

    const int Share_M = TM * BLOCK_X ; 
    const int Share_N = TN * BLOCK_Y ; 
    const int Share_K = 8;              
    const int test_times = 10;         // kernel test times

    dim3 block(BLOCK_X, BLOCK_Y); 
    dim3 grid((M + block.x - 1) / block.x,  
                (N + block.y - 1) / block.y); // Grid size
    
    float *a,*b,*c;
    float *A,*B,*C;
    // Allocate host memory
    a = (float *)malloc(M*K*sizeof(float));
    b = (float *)malloc(K*N*sizeof(float));
    c = (float *)malloc(M*N*sizeof(float));
    // Allocate device memory
    cudaMalloc((void **)&A,M*K*sizeof(float));
    cudaMalloc((void **)&B,K*N*sizeof(float));
    cudaMalloc((void **)&C,M*N*sizeof(float));
   
    // Initialize host memory
    for(int i=0;i<M*K;i++){
        a[i] = 1;
    }
    for(int i=0;i<K*N;i++){
        b[i] = 1;
    }
    memset(c,0,M*N*sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(A,a,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B,b,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(C,c,M*N*sizeof(float),cudaMemcpyHostToDevice);


    /*-----------------------Calculate time -------------------------*/
    
    // GPU warm-up 
    kernel_mul_v3<Share_M, Share_N, Share_K, TM, TN><<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();  // wait for warm-up to finish
    
    //todo kernel time
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    float kernel_total_time = 0.0f;
    
    cudaEventRecord(kernel_start,0);  // start timing
    
    for(int i=0; i<test_times; i++) 
    {
        kernel_mul_v3<Share_M, Share_N, Share_K, TM, TN><<<grid, block>>>(A, B, C, M, K, N);  // launch kernel
    }
    
    cudaEventRecord(kernel_end,0);      // end timing
    cudaEventSynchronize(kernel_end); // wait for GPU to finish
    cudaEventElapsedTime(&kernel_total_time, kernel_start, kernel_end); // must float ker_time
    float kernel_avg_time = kernel_total_time / test_times;  // average time (milliseconds)
    
    //todo CPU time
    double cpu_total_time = 0.0;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    // for(int i=0; i<test_times; i++) {
    cpu_matrix_mul(a, b, c, M, K, N); 
    // }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_elapsed = cpu_end - cpu_start;
    cpu_total_time = cpu_elapsed.count();
    double cpu_avg_time = cpu_total_time;  // CPU average time (milliseconds)
    

    float *C_Device = (float *)malloc(M*N*sizeof(float));
    cudaMemcpy(C_Device,C,M*N*sizeof(float),cudaMemcpyDeviceToHost);
    verify_result(c, C_Device, M*N);
    
    printf("(M:%d,K:%d,N:%d)\n",M,K,N);
    printf("Grid:(%d, %d, %d)\nBlock:(%d, %d, %d)\n",grid.x,grid.y,grid.z,block.x,block.y,block.z);
    
    printf("Kernel Time: %f ms\n", kernel_avg_time);
    printf("CPU Time: %f ms\n", cpu_avg_time);
    // Free device memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    // Free host memory
    free(a);
    free(b);
    free(c);


    return 0;
}

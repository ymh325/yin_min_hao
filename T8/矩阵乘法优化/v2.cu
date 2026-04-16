#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

// Not divided into sections
template <int BLOCK> // block size = BLOCK^2 
__global__ void kernel_mul(float *A, float *B, float *C, int M, int K, int N)
{ 
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ float SA[BLOCK][32];
    __shared__ float SB[32][BLOCK];
    if(row >= M || col >= N){
        return;
    }
    for(int id = threadIdx.y; id < K; id+=BLOCK){
        SA[threadIdx.x][id] = A[row*K+id];
    }
    for(int id = threadIdx.x; id < K; id+=BLOCK){
        SB[id][threadIdx.y] = B[id*N+col];
    }
    __syncthreads();
    float tmp = 0;
    for(int id = 0; id < K; id++){
        tmp += SA[threadIdx.x][id] * SB[id][threadIdx.y];
    }
    C[row * N + col] = tmp;
    __syncthreads();
}

// Divided into sections
template <int BLOCK_DIM> // block size = BLOCK_DIM^2
__global__ void kernel_mul_v2(float *dA, float *dB, float *dC, int M, int K, int N){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    float tmp = 0.0f;
    __shared__ float SA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float SB[BLOCK_DIM][BLOCK_DIM];
    int width = (K + BLOCK_DIM - 1) / BLOCK_DIM;
    for (int ph = 0; ph < width; ph++){
        if (row < M && threadIdx.y + ph * BLOCK_DIM < K){
            SA[threadIdx.x][threadIdx.y] = dA[row * K + threadIdx.y + ph * BLOCK_DIM];
        }
        else{
            SA[threadIdx.x][threadIdx.y] = 0.0f;
        }
        if (col < N && threadIdx.x + ph * BLOCK_DIM < K){
            SB[threadIdx.x][threadIdx.y] = dB[(threadIdx.x + ph * BLOCK_DIM) * N + col];
        }
        else{
            SB[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int s = 0; s < BLOCK_DIM; s++){
            tmp += SA[threadIdx.x][s] * SB[s][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        dC[row * N + col] = tmp;
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
    const int M = 1024;   
    const int K = 1024;   
    const int N = 1024;   
    const int test_times = 10;

    dim3 block_z(32, 32);
    dim3 grid_z((M + block_z.x - 1) / block_z.x,  
                (N + block_z.y - 1) / block_z.y); 
    
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
    kernel_mul_v2<32><<<grid_z, block_z>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();  // wait for warm-up to finish
    
    //todo kernel time
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    float kernel_total_time = 0.0f;
    
    cudaEventRecord(kernel_start,0);  // start timing
    
    for(int i=0; i<test_times; i++) 
    {
        kernel_mul_v2<32><<<grid_z, block_z>>>(A, B, C, M, K, N);  // launch kernel
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
    printf("Grid:(%d, %d, %d)\nBlock:(%d, %d, %d)\n",grid_z.x,grid_z.y,grid_z.z,block_z.x,block_z.y,block_z.z);
    
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

#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
template<int SM,int SN,int SK,int TM,int TN>
__global__ void kernel_mul_v4(float *dA,float *dB,float*dC,int M,int K,int N){
    __shared__ float SA[SM*SK];
    __shared__ float SB[SK*SN];
    int inA = TM * (blockIdx.x * blockDim.x);
    int inB = TN * (blockIdx.y * blockDim.y);
    int id = threadIdx.x + threadIdx.y * blockDim.x;
    
    int SA_x = id % 128; 
    int SA_y = id / 128; 
    int SB_x = id % 8;
    int SB_y = id / 8;

    int with = (K + SK - 1)/SK;
    float tmp[TM * TN] = {0.0f};

    for(int p=0;p<with;p++){
        if(inA + SA_x < M && p * SK +SA_y < K ){
            SA[SA_x * SK + SA_y] = dA[(inA + SA_x) * K + p * SK + SA_y ];
        }
        else {
            SA[SA_x * SK + SA_y] = 0.0f;
        }
        if(p * SK + SB_x < K && inB + SB_y < N){
            SB[SB_x * SN + SB_y] = dB[(p * SK + SB_x) * N + inB + SB_y];
        }
        else {
            SB[SB_x * SN + SB_y] = 0.0f;
        }
        __syncthreads();
        
        for(int row = 0 ; row < TM; row++){
            for(int col = 0;col < TN;col++){
                int  i = row + threadIdx.x * TM;
                int  j = col + threadIdx.y * TN;
                for(int k = 0; k < SK ; k++){
                    tmp[row * TN + col] += SA[i * SK + k] * SB[k * SN + j];                    
                }
            }
        }
        __syncthreads();
    }
    for(int row = 0 ; row < TM; row++){
        for(int col = 0;col < TN;col++){
            int  i =inA + row + threadIdx.x * TM;
            int  j =inB + col + threadIdx.y * TN ;
            if(i < M && j < N){
                dC[i * N + j] = tmp[row * TN + col];                    
            }
        }
    }
}

void cpu_matrix_mul(float *a, float *b, float *c, int M, int K, int N) {
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float sum = 0;
            for(int k=0;k<K;k++){
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
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

    const int SM = TM * BLOCK_X ; 
    const int SN = TN * BLOCK_Y ; 
    const int SK = 8;              
    const int test_times = 10;         // kernel test times

    int num_blocks_x = (M + SM - 1) / SM;
    int num_blocks_y = (N + SN - 1) / SN;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(num_blocks_x, num_blocks_y);
    // dim3 grid((M + block.x - 1) / block.x,  
    //             (N + block.y - 1) / block.y); // Grid size
    
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
    kernel_mul_v4<SM, SN, SK, TM, TN><<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();  // wait for warm-up to finish
    
    //todo kernel time
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    float kernel_total_time = 0.0f;
    
    cudaEventRecord(kernel_start,0);  // start timing
    
    for(int i=0; i<test_times; i++) 
    {
        kernel_mul_v4<SM, SN, SK, TM, TN><<<grid, block>>>(A, B, C, M, K, N);  // launch kernel
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

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define int size_t

const int N = 10240; // aixs - X
const int M = 10240; // aixs - Y
const int nn = N + 2; 
const int mm = M + 2; 

const int block_X = 16;
const int block_Y = 16;

// const int grid_X = (N + block_X-2 - 1) / (block_X-2);
// const int grid_Y = (M + block_Y-2 - 1) / (block_Y-2);

const int tile_x = block_X - 2; // interior count per block in x
const int tile_y = block_Y - 2;
const int grid_X = (M + tile_x - 1) / tile_x; // columns
const int grid_Y = (N + tile_y - 1) / tile_y; // rows


__global__ void DeviceMul(float *dA ,float *dAnew){
    int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if(i <= N  && j <= M){
        dAnew[i*mm + j] = 0.25 * (dA[(i-1)*mm + j] + dA[(i+1)*mm + j] + dA[i*mm + (j-1)] + dA[i*mm + (j+1)]);
    }
}

// __global__ void DeviceMul2th(float *dA ,float *dAnew){
//     int i = threadIdx.y + blockIdx.y * blockDim.y + 1;
//     int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
//     int thx = threadIdx.x + 1;
//     int thy = threadIdx.y + 1;
//     int lenx = blockDim.x + 2;
//     int leny = blockDim.y + 2;
//     __shared__ float SA[(block_X+2)*(block_Y+2)];
//     if(i<=N && j<=M){
//         SA[thy * lenx + thx] = dA[i * mm + j];
//         if(thx == 1)SA[thy * lenx] = dA[i * mm + j - 1];
//         if(thx == lenx - 2)SA[thy  * lenx + thx + 1] = dA[i * mm + j + 1];
//         if(thy == 1)SA[thx] = dA[(i-1) * mm + j];
//         if(thy == leny - 2)SA[(thy+1) * lenx + thx] = dA[(i+1) * mm + j];
//     }
//     else  SA[thy * lenx + thx] = 0.0f;
//     __syncthreads();
//     if(i<=N && j<=M){
//         dAnew[i * mm + j] = 0.25 * (SA[thy * lenx + thx-1] + SA[thy * lenx + thx + 1] + SA[(thy-1) * lenx + thx] + SA[(thy+1)*lenx + thx]);
//     }
// }
__global__ void DeviceMul2th(float *dA, float *dAnew) {
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int lenx = blockDim.x;
    int leny = blockDim.y;
    int i = blockIdx.y * (blockDim.y-2) + ty ;
    int j = blockIdx.x * (blockDim.x-2) + tx ;
    __shared__ float SA[(block_Y)*(block_X)];
    if(i < nn && j < mm){
        SA[ty * lenx + tx] =  dA[i * mm + j];
    }
    else SA[ty * lenx + tx] =0.0f;

    __syncthreads();
    if(1 <= tx && tx < lenx-1 && 1<=ty && ty < leny - 1 ){
        if(i > 0 && i <= N && j > 0 && j <= M)
        dAnew[i * mm + j] = 0.25f * (SA[ty * lenx + tx-1] + SA[ty * lenx + tx + 1] +
                                     SA[(ty-1) * lenx + tx] + SA[(ty+1)*lenx + tx]);
    }
}

void HostMul(float *dA,float *dAnew){
    for(int i=1;i<=N;i++){
        for(int j=1;j<=M;j++){
            dAnew[i * mm + j] = 0.25 * (dA[(i-1)*mm + j] + dA[(i+1)*mm + j] + dA[i*mm + (j-1)] + dA[i*mm + (j+1)]);
        }
    }
}

void PASS(float *host, float *dev , float eps = 1e-5){
    for(int i=1;i<=N;i++){
        for(int j=1;j<=M;j++){
            int id = i * mm + j;
            if(fabs(host[id] - dev[id]) > 1e-5){
                printf("error host[%zu][%zu] = %f, dev[%zu][%zu] = %f\n",i,j,host[id],i,j,dev[id]);
                exit(1);
            }
        }
    }
    printf("PASS\n");
} 


signed main(){
    const int times = 1;
    const size_t size = nn*mm*sizeof(float);

    dim3 block(block_X, block_Y);
    dim3 grid(grid_X, grid_Y);
    float *hA, *hAnew, *dA, *dAnew;
    // Allocate host and device 
    hA = (float *)malloc(size);
    hAnew = (float *)malloc(size);

    cudaMalloc(&dA, size);
    cudaMalloc(&dAnew, size);

    // Initialize host 
    memset(hA,0,size);
    memset(hAnew,0,size);
    for(int i=1;i<=N;i++){
        for(int j=1;j<=M;j++){
            hA[i*mm + j] = 1;
        }
    }

    // Copy host to device 
    cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dAnew,hAnew,size,cudaMemcpyHostToDevice);

    // GPU run
    float Device_time = 0.0f;
    cudaEvent_t Device_start, Device_end;
    cudaEventCreate(&Device_start);
    cudaEventCreate(&Device_end);
    cudaEventRecord(Device_start,0);
    for(int i=1;i<=times;i++){
        DeviceMul<<<grid,block>>>(dA,dAnew);
        float *tmp = dA;
        if(i<times){
            dA = dAnew;
            dAnew = tmp;
        }
    }
    cudaEventRecord(Device_end,0);
    cudaEventSynchronize(Device_end);
    cudaEventElapsedTime(&Device_time, Device_start, Device_end);
    float Device_avg = Device_time / times;
    
    cudaDeviceSynchronize();

    printf("times: %zu\n",times);
    printf("Device_time: %f ms  avg: %f\n", Device_time,Device_avg);
    
    // Free host and device 
    free(hA);
    free(hAnew);
    cudaFree(dA);
    cudaFree(dAnew);
    return 0;
}
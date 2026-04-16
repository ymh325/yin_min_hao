#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

const int N = 10240; // aixs - col
const int M = 10240; // aixs - row
const int nn = N + 2;  // row
const int mm = M + 2;  // col

const int block_X = 32;
const int block_Y = 8;

// const int grid_X = (M + block_X - 1) / (block_X); // col
// const int grid_Y = (N + block_Y - 1) / (block_Y); // row

const int grid_X = (M + block_X-2 - 1) / (block_X-2); // col
const int grid_Y = (N + block_Y-2 - 1) / (block_Y-2); // row

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
    int tx = threadIdx.x, ty = threadIdx.y;
    int lenx = blockDim.x, leny = blockDim.y;

    int i = blockIdx.y * (leny - 2) + ty;
    int j = blockIdx.x * (lenx - 2) + tx;

    __shared__ float SA[block_Y][block_X + 1];

    if (i < nn && j < mm)
        SA[ty][tx] = dA[i * mm + j];
    else
        SA[ty][tx] = 0.0f;

    __syncthreads();
    if (tx >= 1 && tx < lenx - 1 &&
        ty >= 1 && ty < leny - 1 &&
        i > 0 && i <= N &&
        j > 0 && j <= M)
    {
        dAnew[i * mm + j] =
            0.25f * (
                SA[ty][tx - 1] +
                SA[ty][tx + 1] +
                SA[ty - 1][tx] +
                SA[ty + 1][tx]
            );
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
                printf("error host[%d][%d] = %f, dev[%d][%d] = %f\n",i,j,host[id],i,j,dev[id]);
                exit(1);
            }
        }
    }
    printf("PASS\n");
} 


int main(){

    const int times = 200;
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
        DeviceMul2th<<<grid,block>>>(dA,dAnew);
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
    
    
    double Host_time = 0.0;
    auto Host_start = std::chrono::high_resolution_clock::now();
    for(int i=1;i<=times;i++){
        HostMul(hA,hAnew);
        float *tmp = hA;
        if(i<times){
            hA = hAnew;
            hAnew = tmp;
        }
    }
     auto Host_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> Host_elapsed = Host_end - Host_start;
    Host_time = Host_elapsed.count();
    float Host_avg = Host_time / times;
    cudaDeviceSynchronize();
    
    // Copy device to host 
    float *dAnew_host = (float *)malloc(size);
    cudaMemcpy(dAnew_host,dAnew,size,cudaMemcpyDeviceToHost);
    
    // test result
    PASS(hAnew,dAnew_host);
    
    printf("times: %d\n",times);
    printf("Device_time: %f ms  avg: %f\n", Device_time,Device_avg);
    printf("Host_time: %f ms avg: %f\n", Host_time,Host_avg);
    
    // Free host and device 
    free(hA);
    free(hAnew);
    free(dAnew_host);
    cudaFree(dA);
    cudaFree(dAnew);
    return 0;
}
#include <iostream>
#include "Module.cuh"
using namespace std;
__global__ void call_mesh2d(int rank,int n, int m, double* x, double* y,int pre)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y+1;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i <= ni ){ 
        x[i] = xa + (i-1) * dx;
    }

    if(j < nj){
        y[j] = ya + j * dy;
    }
    
    return ;
}
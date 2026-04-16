#include <iostream>
#include "Module.cuh"
__global__ void call_initia(int rank,int n, int m , double* f, double* fm1, double* fm2, double* x, double* y,int pre)
{
    // 计算线程对应的i和j索引
    int local_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int global_i = local_i + pre;

    // 边界检查
    if (global_i > ni || local_i > n || j >= nj) {
        return;
    }
    
    int idx = local_i * m + j;
    
    f[idx] = 0.0;
    fm1[idx] = 0.0;
    fm2[idx] = 0.0;
    
    // 设置初始条件（用 global_i 查全局坐标数组）
    if (0.2 <= x[global_i] && x[global_i] <= 0.5 && 0.2 <= y[j] && y[j] <= 0.5)
    {
        f[idx] = 1;
    }
    return ;
}
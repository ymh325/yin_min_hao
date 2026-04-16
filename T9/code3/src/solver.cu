#include <iostream>
#include "Module.cuh"

// 第一个 kernel：计算 fm1
__global__ void compute_fm1(int ni, int nj, int n, double* f, double* fm1, double dt, int pre) {
    int local_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int global_i = local_i + pre;

    if (global_i <= 1 || global_i > ni || local_i > n || j < 1 || j >= nj) return;

    int idx = local_i * nj + j;
    int idx_im1 = (local_i - 1) * nj + j;
    double f_ij = f[idx];
    fm1[idx] = z * dt * f_ij * (1 - f_ij) * (f_ij - 0.5)
               + U * dt / dx * (f[idx_im1] - f_ij)
               + f_ij;
}

// 第二个 kernel：计算 fm2
__global__ void compute_fm2(int ni, int nj, int n, double* fm1, double* fm2, double dt, int pre) {
    int local_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int global_i = local_i + pre;

    if (global_i < 1 || global_i > ni || local_i > n || j < 1 || j >= nj) return;

    int idx = local_i * nj + j;
    int idx_jm1 = local_i * nj + (j - 1);
    double fm1_ij = fm1[idx];
    fm2[idx] = z * dt * fm1_ij * (1 - fm1_ij) * (fm1_ij - 0.5)
               + V * dt / dy * (fm1[idx_jm1] - fm1_ij)
               + fm1_ij;
}

// 第三个 kernel：更新 f
__global__ void update_f(int ni, int nj, int n, double* f, double* fm2, int pre) {
    int local_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int global_i = local_i + pre;

    if (global_i < 1 || global_i > ni || local_i > n || j < 1 || j >= nj) return;

    int idx = local_i * nj + j;
    f[idx] = fm2[idx];
}

// 主机端调用函数
void call_solver(int ni, int nj, int n, double* f, double* fm1, double* fm2, double dt, int pre, dim3 grid, dim3 block) {

    compute_fm1<<<grid, block>>>(ni, nj, n, f, fm1, dt, pre);
    compute_fm2<<<grid, block>>>(ni, nj, n, fm1, fm2, dt, pre);
    update_f<<<grid, block>>>(ni, nj, n, f, fm2, pre);
}
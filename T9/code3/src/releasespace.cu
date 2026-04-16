#include <iostream>
#include <cuda_runtime.h>
#include "Module.cuh"
void call_releasespace()
{
    cudaFree(x);
    cudaFree(y);
    cudaFree(f);
    cudaFree(fm1);
    cudaFree(fm2);
    return;
}
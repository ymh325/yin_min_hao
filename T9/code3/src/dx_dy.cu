#include "Module.cuh"

__global__ void dx_dy()
{
    dx = (xb - xa) / (ni - 1);
    dy = (yb - ya) / (nj - 1);
    return ;
}
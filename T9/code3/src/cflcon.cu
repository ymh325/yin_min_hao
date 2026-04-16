#include <iostream>
#include "Module.cuh"
using namespace std;
__global__ void call_cflcon()
{
    // if(dx==0||dy==0){
    //     printf("Error dx==0||dy==0\n");
    //     return ;
    // }
    dtx = cfl * dx / U;
    dty = cfl * dy / V;
    // printf("dtx = %f, dty = %f\n", dtx, dty);
    dt_device = min(dtx, dty);
    return;
}
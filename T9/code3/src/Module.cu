//----------------------------------------------------------------------------------------
#include <cuda_runtime.h>

__managed__ int ni = 20001;
__managed__ int nj = 20001;                                                      
double tout = 0.2;
double* x = nullptr;
double* y = nullptr;
double* f = nullptr;    // contiguous array of size ni*nj
double* fm1 = nullptr;  // contiguous array of size ni*nj
double* fm2 = nullptr;  // contiguous array of size ni*nj
double time0;
__device__ double xa = 0.0;                                                                         //
__device__ double xb = 1.0;                                                                          //
__device__ double ya = 0.0;
__device__ double yb = 1.0;
__device__ double cfl = 0.8;
__device__ double U = 1.0;
__device__ double V = 2.0;
__device__ double z = 100;
__device__ double dx;
__device__ double dy;
__device__ double dtx;
__device__ double dty;
double dt_host; // host
__device__ double dt_device; // device

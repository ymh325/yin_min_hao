//----------------------------------------------
#pragma once
extern __managed__ int ni;
extern __managed__ int nj;
extern  double tout;
extern  double* x;
extern  double* y;
extern  double* f;
extern  double* fm1;
extern  double* fm2;
extern double time0;
extern __device__ double xa;
extern __device__ double xb;
extern __device__ double ya;
extern __device__ double yb;
extern __device__ double dx;
extern __device__ double dy;
extern __device__ double z;           
extern __device__ double U;
extern __device__ double V;
extern __device__ double cfl;
extern __device__ double dtx;
extern __device__ double dty;
extern double dt_host;
extern __device__ double dt_device;

#ifndef JACOBI_H
#define JACOBI_H
  
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>


void Initial(float *&dA, float *&dAnew,int n,int m);
void Lanuch_kernel(float *dA, float *dAnew, int n,int m);
void Free(float *dA, float *dAnew);

#endif
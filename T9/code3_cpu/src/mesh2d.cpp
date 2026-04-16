#include <iostream>
#include "Module.h"
using namespace std;

void call_mesh2d(int rank, int n, int m, double* x, double* y, int pre)
{
    for (int i = 1; i <= ni; i++) {
        x[i] = xa + (i - 1) * dx;
    }

    for (int j = 0; j < nj; j++) {
        y[j] = ya + j * dy;
    }

    return;
}

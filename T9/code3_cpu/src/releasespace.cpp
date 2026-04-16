#include <iostream>
#include "Module.h"

void call_releasespace()
{
    delete[] x;
    delete[] y;
    delete[] f;
    delete[] fm1;
    delete[] fm2;
    return;
}

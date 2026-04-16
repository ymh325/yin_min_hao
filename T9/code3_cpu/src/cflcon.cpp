#include <iostream>
#include <algorithm>
#include "Module.h"
using namespace std;

void call_cflcon()
{
    dtx = cfl * dx / U;
    dty = cfl * dy / V;
    dt = min(dtx, dty);
    return;
}

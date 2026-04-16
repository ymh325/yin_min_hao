//----------------------------------------------------------------------------------------

int ni = 20001;
int nj = 20001;
double tout = 0.2;
double* x = nullptr;
double* y = nullptr;
double* f = nullptr;    // contiguous array of size ni*nj
double* fm1 = nullptr;  // contiguous array of size ni*nj
double* fm2 = nullptr;  // contiguous array of size ni*nj
double time0;
double xa = 0.0;
double xb = 1.0;
double ya = 0.0;
double yb = 1.0;
double cfl = 0.8;
double U = 1.0;
double V = 2.0;
double z = 100;
double dx;
double dy;
double dtx;
double dty;
double dt;

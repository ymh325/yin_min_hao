//----------------------------------------------------------------

//----------------------------------------------------------------
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "Module.h"

void call_dx_dy();
void call_mesh2d(int rank, int n, int m, double* x, double* y, int pre);
void call_initia(int rank, int n, int m, double* f, double* fm1, double* fm2, double* x, double* y, int pre);
void call_cflcon();
void call_bounda();
void call_solver(int ni, int nj, int n, double* f, double* fm1, double* fm2, double dt, int pre);
void call_output(double num, int rank, int ni, int nj, double* x, double* y, double* f);
void call_releasespace();

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = ni / size;
    int m = nj;
    int pre = rank * ni / size; // 前面有多少行数据
    if (rank == size - 1) {
        n += ni % size;
    }
    printf("rank = %d, n = %d, m = %d\n", rank, n, m);

    call_dx_dy();

    x = new double[(ni + 1)];
    y = new double[nj];
    f = new double[(n + 2) * m];
    fm1 = new double[(n + 2) * m];
    fm2 = new double[(n + 2) * m];

    // 初始化为0
    memset(f, 0, (n + 2) * m * sizeof(double));
    memset(fm1, 0, (n + 2) * m * sizeof(double));
    memset(fm2, 0, (n + 2) * m * sizeof(double));

    call_mesh2d(rank, n, m, x, y, pre);

    call_initia(rank, n, m, f, fm1, fm2, x, y, pre);

    call_cflcon();

    // host
    time0 = 0;

    if (dt <= 0) {
        printf("Error dt = %f , rank = %d\n", dt, rank);
        exit(1);
    }

    printf("--------------Calculated dt = %f , rank = %d-----------------\n", dt, rank);

    int total_steps = (int)(tout / dt);
    int output_interval = total_steps / 100;
    if (output_interval < 1) output_interval = 1;
    int output_count = 0;

    if (rank == 0) {
        printf("Total time steps: %d, Output interval: %d steps\n", total_steps, output_interval);
    }

    // 使用 MPI_Wtime 计时
    double t_start = MPI_Wtime();

    int out = 0;
    int iteration = 0;
    do {
        MPI_Request request;
        if (rank < size - 1) {
            MPI_Isend(f + n * m, m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
        }
        if (rank > 0) {
            MPI_Recv(f, m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }

        call_solver(ni, nj, n, f, fm1, fm2, dt, pre);

        time0 = time0 + dt;

        iteration++;
        // 每 output_interval 步或最后一步输出一次，总共约 100 个文件
        if (out && (iteration % output_interval == 0 || time0 > tout)) {
            output_count++;
            double* h_x = NULL;
            double* h_y = NULL;
            double* h_f = NULL;

            // Only rank 0 needs to allocate full arrays
            if (rank == 0) {
                h_x = new double[ni];
                h_y = new double[nj];
                h_f = new double[ni * nj];

                // rank 0 copies data
                memcpy(h_x, x + 1, ni * sizeof(double));
                memcpy(h_y, y, nj * sizeof(double));
                memcpy(h_f, f + 1 * m, n * m * sizeof(double));
                // receive data from other ranks
                if (size > 1) {
                    MPI_Request* requests = new MPI_Request[size - 1];
                    for (int ran = 1; ran < size; ran++) {
                        int i = ni / size; // 获取当前 rank 的数据行数
                        int j = nj;
                        if (ran == size - 1) {
                            i += ni % size;
                        }
                        MPI_Irecv(h_f + (ran * ni / size) * j, i * j, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD, requests + ran - 1);
                    }
                    MPI_Waitall(size - 1, requests, MPI_STATUSES_IGNORE);
                    delete[] requests;
                }
            }
            else {
                // ! may be wrong
                // 给 h_f 分配内存
                h_f = new double[n * m];
                // 拷贝数据
                memcpy(h_f, f + 1 * m, n * m * sizeof(double));
                // 发送数据到 rank 0
                MPI_Request req;
                MPI_Isend(h_f, n * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
                MPI_Wait(&req, MPI_STATUS_IGNORE);
                delete[] h_f;
            }

            if (rank == 0) {
                call_output(output_count, rank, ni, nj, h_x, h_y, h_f);
            }
        }

        if(rank==0)printf("Output written for time step %d\n", iteration);
    } while (time0 <= tout);

    // 计算经过的时间
    double t_end = MPI_Wtime();
    double elapsedTime = t_end - t_start;

    printf("====================================================\n");
    printf("rank = %d, n = %d, m = %d\n", rank, n, m);
    printf("Total time: %.4f s\n", elapsedTime);
    printf("Average time per time step (CPU): %.4f s\n", elapsedTime / iteration);

    if (rank == 0) {
        printf("Total output files: %d\n", output_count);
    }

    call_releasespace();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}

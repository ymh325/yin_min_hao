#include "Jacobi.h"

const int total_n = 20000; // Jacobi Size
const int times = 10;

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < size) {
        if (rank == 0) {
            printf("Warning: Only %d GPUs available for %d MPI processes\n", 
                   num_gpus, size);
            exit(1);
        }
    }
    cudaSetDevice(rank % num_gpus);

    printf("rank %d, size %d\n", rank, size);
    // Divide the entire grid horizontally into [(total_n + size - 1) / szie] parts.
    int n = 0,m = total_n + 2;
    if(rank==size-1)n = total_n / size + total_n % size;
    else  n = total_n / size;

    n += 2;

    float *dA;
    float *dAnew;
    Initial(dA, dAnew, n , m);
    
    int top = (rank == 0 ? MPI_PROC_NULL : rank - 1);
    int bottom = (rank == size - 1 ? MPI_PROC_NULL : rank + 1);

    MPI_Barrier(MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    for(int i = 0; i < times; i++){
        MPI_Request reqs[4];
        // 启动所有接收
        MPI_Irecv(dA + (n-1)*m, m, MPI_FLOAT, bottom, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(dA,           m, MPI_FLOAT, top,    0, MPI_COMM_WORLD, &reqs[1]);

        // 启动所有发送
        MPI_Isend(dA + (n-2)*m, m, MPI_FLOAT, bottom, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(dA + m,       m, MPI_FLOAT, top,    0, MPI_COMM_WORLD, &reqs[3]);

        // 等待所有完成
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        Lanuch_kernel(dA, dAnew, n, m);
        cudaDeviceSynchronize();
        float *tmp = dA;
        dA = dAnew;
        dAnew = tmp;
    }

    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double end_time = MPI_Wtime();
    
    Free(dA, dAnew);
    if (rank == 0) {
        printf("Total iterations: %d\n", times);
        printf("Total execution time: %.4f s\n", end_time - start_time);
        printf("Average time per iteration: %.4f s\n", (end_time - start_time) / times);
    }

    // if(rank == 0){
    //     float *tm = (float*)malloc((n + 2) * m * sizeof(float));
    //     cudaMemcpy(tm, dA, (n + 2) * m * sizeof(float), cudaMemcpyDeviceToHost);
    //     for(int i = 0; i < n+2; i++){
    //         for(int j = 0; j < m; j++){
    //             printf("%f ", tm[i * m + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     free(tm);
    // }

    
    MPI_Finalize();
    return 0;
}

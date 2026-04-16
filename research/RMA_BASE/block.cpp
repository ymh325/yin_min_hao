#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "var.h"

#define idx(row, col) ((row) * (m + 2) + (col))
int main(int argc,char *argv[]){
    int size,rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int m = M;
    int n = N/size;
    if(rank == size-1){
        n += N%size;
    }
    
    //-------------------------------
    float *a = NULL;
    float *b = NULL;
    size_t num = (n+2)*(m+2);
    a = (float*)calloc(num,sizeof(float));
    b = (float*)calloc(num,sizeof(float));

    if(a == NULL){
        fprintf(stderr,"rank %d malloc failed\n",rank);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            a[idx(i,j)] = 1.0f;
        }
    }
    
    double start, end;
    double comm_time = 0.0, calc_time = 0.0;
    start = MPI_Wtime();
    //-------------------------------
    for(int t=1;t<=times;t++){
        double t1 = MPI_Wtime();
        if(rank > 0){
            MPI_Sendrecv(&a[idx(1,1)],m,MPI_FLOAT,rank-1,0,
                         &a[idx(0,1)],m,MPI_FLOAT,rank-1,1,
                         MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        if(rank < size-1){
            MPI_Sendrecv(&a[idx(n,1)],m,MPI_FLOAT,rank+1,1,
                         &a[idx(n+1,1)],m,MPI_FLOAT,rank+1,0,
                         MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        comm_time += MPI_Wtime() - t1;

        double t2 = MPI_Wtime();
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                double tem = 0.0f;int cnt=0;
                if(i>=1)tem += a[idx(i-1,j)],cnt++;
                if(j-1>=1)tem += a[idx(i,j-1)],cnt++;
                if(i<=n)tem += a[idx(i+1,j)],cnt++;
                if(j+1<=m)tem += a[idx(i,j+1)],cnt++;
                b[idx(i,j)] = tem/cnt;
            }
        }
        calc_time += MPI_Wtime() - t2;
        float* x;
        x = a;
        a = b;
        b = x;
        if(rank==0){
            printf("time = %d\n",t);
        }
    }

    //-------------------------------
    end = MPI_Wtime();
    printf("rank %d time: %.4f s, comm time: %.4f s, calc time: %.4f s\n", rank, end - start, comm_time, calc_time);

    free(a);
    free(b);
    MPI_Finalize();

    return 0;
}
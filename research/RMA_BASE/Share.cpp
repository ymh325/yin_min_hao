#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "var.h"

#define idx(r,c) ((r)*(m+2)+(c))

int main(int argc,char *argv[]){

    MPI_Init(&argc,&argv);

    int world_rank,world_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    /* Divide communicator by node */
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        0,
                        MPI_INFO_NULL,
                        &node_comm);

    int rank,size;
    MPI_Comm_rank(node_comm,&rank);
    MPI_Comm_size(node_comm,&size);

    int m=M;
    int n=N/size;
    if(rank==size-1) n+=N%size;

    size_t num=(n+2)*(m+2);

    float *a;
    float *b=(float*)calloc(num,sizeof(float));

    MPI_Win win;

    /* Allocate shared memory */
    MPI_Win_allocate_shared(
        num*sizeof(float),
        sizeof(float),
        MPI_INFO_NULL,
        node_comm,
        &a,
        &win
    );


    float *up_ptr=NULL;
    float *down_ptr=NULL;

    MPI_Aint size_query;
    int disp;

    // query up neighbor
    if(rank>0)
        MPI_Win_shared_query(win,rank-1,&size_query,&disp,&up_ptr);

    // query down neighbor
    if(rank<size-1)
        MPI_Win_shared_query(win,rank+1,&size_query,&disp,&down_ptr);

    /* inital */
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
        a[idx(i,j)] = 1.0;

    double start=MPI_Wtime();
    double comm_time = 0.0, calc_time = 0.0;

    for(int t=0;t<times;t++){

        double t1 = MPI_Wtime();
        /* halo  */
        if(rank>0)
            for(int j=1;j<=m;j++)
                a[idx(0,j)] = up_ptr[idx(n,j)];

        if(rank<size-1)
            for(int j=1;j<=m;j++)
                a[idx(n+1,j)] = down_ptr[idx(1,j)];

        MPI_Win_sync(win);
        comm_time += MPI_Wtime() - t1;

        double t2 = MPI_Wtime();
        for(int i=1;i<=n;i++)
            for(int j=1;j<=m;j++){

                float s =
                    a[idx(i-1,j)] +
                    a[idx(i+1,j)] +
                    a[idx(i,j-1)] +
                    a[idx(i,j+1)];
                b[idx(i,j)] = 0.25f*s;
            }
        calc_time += MPI_Wtime() - t2;

        float *x=a;
        a=b;
        b=x;
        if(rank==0){
            printf("time = %d\n",t);
        }
    }

    double end=MPI_Wtime();

    printf("rank %d time %.4f comm time %.4f calc time %.4f\n",world_rank,end-start,comm_time,calc_time);

    MPI_Win_free(&win);

    MPI_Finalize();
}
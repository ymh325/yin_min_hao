#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "var.h"

#define idx(row, col) ((row) * (m + 2) + (col))

int main(int argc,char *argv[]){
    MPI_Init(&argc,&argv);

    int size,rank;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank==0) printf("size = %d , Time = %d , CPU = %d\n",N,iters,size);


    int m = M;
    int nn = N/size;
    int n = nn;
    if(rank == size-1) n += N%size;


    //---------------------------------------
    float *a=NULL,*b=NULL;
    size_t num = (n+2)*(m+2);

    a=(float*)calloc(num,sizeof(float));
    b=(float*)calloc(num,sizeof(float));

    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            a[idx(i,j)] = 1.0f;

    //---------------------------------------
    /* Create RMA window */
    MPI_Win win_a, win_b;
    MPI_Win_create(a, num * sizeof(float), sizeof(float),
                MPI_INFO_NULL, MPI_COMM_WORLD, &win_a);
    MPI_Win_create(b, num * sizeof(float), sizeof(float),
                MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);

    /* world group */
    MPI_Group world;
    MPI_Comm_group(MPI_COMM_WORLD,&world);

    /* neighbor group */
    int ranks[2];
    int cnt=0;
    if(rank>0) ranks[cnt++]=rank-1;
    if(rank<size-1) ranks[cnt++]=rank+1;

    MPI_Group neigh;
    MPI_Group_incl(world,cnt,ranks,&neigh);

    float *cur = a;
    float *next = b;
    
    
    //---------------------------------------
    double start=MPI_Wtime();
    double comm_time = 0.0, calc_time = 0.0;

    for(int t=1;t<=iters;t++){

        double t1 = MPI_Wtime();
        /* target side: expose memory */
        MPI_Win win_cur = (cur==a) ? win_a : win_b;
       


        MPI_Win_post(neigh,0,win_cur);

        /* origin side: start access */
        MPI_Win_start(neigh,0,win_cur);

        /* Read from upper process */
        if(rank>0){
            MPI_Get(&cur[idx(0,1)],m,MPI_FLOAT,
                    rank-1,idx(nn,1),m,MPI_FLOAT,
                    win_cur);
        }

        /* Read from lower process */
        if(rank<size-1){
            MPI_Get(&cur[idx(nn+1,1)],m,MPI_FLOAT,
                    rank+1,idx(1,1),m,MPI_FLOAT,
                    win_cur);
        }

        /* Complete access */
        MPI_Win_complete(win_cur);

        /* Wait for others to access me */
        MPI_Win_wait(win_cur);
        comm_time += MPI_Wtime() - t1;

        //---------------------------------------
        /* Jacobi computation */
        double t2 = MPI_Wtime();

        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                float s =
                    cur[idx(i-1,j)] +
                    cur[idx(i+1,j)] +
                    cur[idx(i,j-1)] +
                    cur[idx(i,j+1)];
                next[idx(i,j)] = 0.25f*s;
            }
        }
        calc_time += MPI_Wtime() - t2;

        float *x=cur;
        cur = next;
        next = x;

        if(rank==0) printf("time = %d\n",t);
    }

    //---------------------------------------
    double end=MPI_Wtime();

    printf("rank %d time %.4f s, comm time %.4f s, calc time %.4f s\n",rank,end-start,comm_time,calc_time);

    MPI_Group_free(&neigh);
    MPI_Group_free(&world);
    MPI_Win_free(&win_a);
    MPI_Win_free(&win_b);
    if(rank==0)printf("-----------------------PSCW---------------------------\n");

    free(a);
    free(b);

    MPI_Finalize();
    return 0;
}
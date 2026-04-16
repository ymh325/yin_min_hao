#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "var.h"

#define idx(r,c) ((r)*(m+2)+(c))

int main(int argc,char *argv[]){

    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int m=M;
    int n=N/size;
    if(rank==size-1) n+=N%size;

    size_t num=(n+2)*(m+2);

    float *a=(float*)calloc(num,sizeof(float));
    float *b=(float*)calloc(num,sizeof(float));

    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            a[idx(i,j)]=1.0;

    MPI_Win win;
    MPI_Win_create(a,num*sizeof(float),sizeof(float),
                   MPI_INFO_NULL,MPI_COMM_WORLD,&win);

    double start=MPI_Wtime();
    double comm_time=0.0, calc_time=0.0;

    for(int t=0;t<times;t++){

        double t1 = MPI_Wtime();
        if(rank>0){

            MPI_Win_lock(MPI_LOCK_SHARED,rank-1,0,win);

            MPI_Get(&a[idx(0,1)],m,MPI_FLOAT,
                    rank-1,idx(n,1),m,MPI_FLOAT,
                    win);

            MPI_Win_unlock(rank-1,win);
        }

        if(rank<size-1){

            MPI_Win_lock(MPI_LOCK_SHARED,rank+1,0,win);

            MPI_Get(&a[idx(n+1,1)],m,MPI_FLOAT,
                    rank+1,idx(1,1),m,MPI_FLOAT,
                    win);

            MPI_Win_unlock(rank+1,win);
        }
        comm_time += MPI_Wtime() - t1;

        double t2 = MPI_Wtime();
        
        // MPI_Barrier(MPI_COMM_WORLD);
        
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                float s =
                    a[idx(i-1,j)] +
                    a[idx(i+1,j)] +
                    a[idx(i,j-1)] +
                    a[idx(i,j+1)];
                b[idx(i,j)] = 0.25f*s;
            }
        }
        calc_time += MPI_Wtime() - t2;

        // MPI_Barrier(MPI_COMM_WORLD);

        float *x=a; a=b; b=x;
        if(rank==0)printf("time=%d\n",t);
    }

    double end=MPI_Wtime();

    printf("rank %d time %.4f comm time %.4f calc time %.4f\n",rank,end-start,comm_time,calc_time);

    MPI_Win_free(&win);

    MPI_Finalize();
}

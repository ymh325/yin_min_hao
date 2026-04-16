#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "var.h"

#define idx(i,j) ((i)*(m+2)+(j))

int main(int argc,char *argv[]){
    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    printf("rank %d start\n",rank);

    int m=M;

    int base=N/size;
    int rem=N%size;

    int n=base+(rank<rem);

    bool has_top = rank>0;
    bool has_bottom = rank<size-1;

    float *a,*b;

    size_t num=(size_t)(n+2)*(m+2);

    a=(float*)calloc(num,sizeof(float));
    b=(float*)calloc(num,sizeof(float));

    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
        a[idx(i,j)]=1.0f;

    MPI_Barrier(MPI_COMM_WORLD);

    double start=MPI_Wtime();
    double comm_time = 0.0, calc_time = 0.0;

    double te = 0;

    for(int t=0;t<times;t++){

        MPI_Request req[4];
        int r=0;
        
        double t1 = MPI_Wtime();

        if(has_top){
            MPI_Irecv(&a[idx(0,1)],m,MPI_FLOAT,rank-1,1,
                      MPI_COMM_WORLD,&req[r++]);

            MPI_Isend(&a[idx(1,1)],m,MPI_FLOAT,rank-1,0,
                      MPI_COMM_WORLD,&req[r++]);
        }

        if(has_bottom){
            MPI_Irecv(&a[idx(n+1,1)],m,MPI_FLOAT,rank+1,0,
                      MPI_COMM_WORLD,&req[r++]);

            MPI_Isend(&a[idx(n,1)],m,MPI_FLOAT,rank+1,1,
                      MPI_COMM_WORLD,&req[r++]);
        }
        comm_time += MPI_Wtime() - t1;

        double t2 = MPI_Wtime();
        /* interior */
        for(int i=2;i<=n-1;i++)
            for(int j=1;j<=m;j++){

                float s =
                    a[idx(i-1,j)] +
                    a[idx(i+1,j)] +
                    a[idx(i,j-1)] +
                    a[idx(i,j+1)];
                b[idx(i,j)] = 0.25f*s;
            }

        calc_time += MPI_Wtime() - t2;
        
        double t3 = MPI_Wtime();
        MPI_Waitall(r,req,MPI_STATUSES_IGNORE);
        comm_time += MPI_Wtime() - t3;
        te += MPI_Wtime() - t3;
        

        /* boundary */

        double t4 = MPI_Wtime();
       
        for(int j=1;j<=m;j++){

            float s1 =
                a[idx(0,j)] +
                a[idx(2,j)] +
                a[idx(1,j-1)] +
                a[idx(1,j+1)];

            b[idx(1,j)] = 0.25f*s1;

        }
        for(int j=1;j<=m;j++){
            float s2 =
                a[idx(n-1,j)] +
                a[idx(n+1,j)] +
                a[idx(n,j-1)] +
                a[idx(n,j+1)];

            b[idx(n,j)] = 0.25f*s2;
        }
        
        calc_time += MPI_Wtime() - t4;

        float *tmp=a;
        a=b;
        b=tmp;

        if(rank==0) printf("time = %d\n",t);
    }

    double end=MPI_Wtime();

    printf("rank %d time %.4f te %.4f comm time %.4f calc time %.4f\n",rank,end-start,te,comm_time,calc_time);

    free(a);
    free(b);

    MPI_Finalize();

    return 0;
}
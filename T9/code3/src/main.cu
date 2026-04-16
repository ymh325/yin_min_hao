//----------------------------------------------------------------

//----------------------------------------------------------------
#include <mpi.h>
#include<iostream>
#include<iomanip>
#include<cuda_runtime.h>
#include"Module.cuh"
__global__ void dx_dy();
__global__ void call_mesh2d(int rank,int n, int m, double* x, double* y, int pre); // todo
__global__ void call_initia(int rank,int n, int m, double* f, double* fm1, double* fm2, double* x, double* y,int pre); // todo
__global__ void call_cflcon(); // todo
__global__ void call_bounda(); // todo
void call_solver(int ni, int nj, int n, double* f, double* fm1, double* fm2,double dt,int pre ,dim3 grid ,dim3 block); // todo
void call_output(double num, int rank, int ni, int nj, double* x, double* y, double* f);
void call_releasespace(); // todo

int main(int argc, char* argv[])
{    
    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    int gpu_id = rank % dev_count;
    cudaSetDevice(gpu_id);
    
    int n = ni/size;    
    int m = nj;
    int pre = rank * ni/size; // 前面有多少行数据
    if(rank==size-1){
        n += ni % size;
    }
    printf("rank = %d, n = %d, m = %d\n", rank, n, m);
    // 创建CUDA事件用于计时
    
    dx_dy<<<1,1>>>();
    cudaDeviceSynchronize();
    
    cudaMallocManaged((void**)&x, (ni+1) * sizeof(double));
    cudaMallocManaged((void**)&y, nj * sizeof(double));
    cudaMallocManaged((void**)&f, (n+2) * m * sizeof(double));
    cudaMallocManaged((void**)&fm1,(n+2) * m * sizeof(double));
    cudaMallocManaged((void**)&fm2, (n+2) * m * sizeof(double));
    
    
	// 配置线程块和网格维度
	dim3 blockDim(16, 16); // 16x16线程块
    const int grid_x= (m + blockDim.x - 1) / blockDim.x;
    const int grid_y= (n + blockDim.y - 1) / blockDim.y;
	dim3 gridDim(grid_x, grid_y); // 计算网格维度
    
    dim3 gridDim2( (nj + blockDim.x - 1) / blockDim.x, (ni + blockDim.y - 1) / blockDim.y);
    
    
    
	call_mesh2d<<<gridDim2,blockDim>>>(rank,n, m, x, y,pre);
	
    cudaDeviceSynchronize();
    
	call_initia<<<gridDim, blockDim>>>(rank,n, m, f, fm1, fm2, x, y, pre);
	
    cudaDeviceSynchronize();
    
	call_cflcon<<<1,1>>>();
    
    cudaDeviceSynchronize();
    
    // host
    time0 = 0;
    dt_host = 0;
    // dt_device => dt_host
    cudaError_t errCopy = cudaMemcpyFromSymbol(&dt_host, dt_device, sizeof(double), 0, cudaMemcpyDeviceToHost);
    if (errCopy != cudaSuccess) {
        printf("CUDA Error (copy dt_device to dt_host): %s\n", cudaGetErrorString(errCopy));
        exit(1);
    }
    
    if(dt_host <= 0){
        printf("Error dt_host = %f , rank = %d\n", dt_host, rank);
        exit(1);
    }
    
    printf("--------------Calculated dt = %f , rank = %d-----------------\n", dt_host , rank);
	
    int total_steps = (int)(tout / dt_host);
    int output_interval = total_steps / 100;
    if (output_interval < 1) output_interval = 1;
    int output_count = 0;
    
    if(rank==0){
       printf("Total time steps: %d, Output interval: %d steps\n", total_steps, output_interval);
    }
    
    cudaEvent_t start, stop;
    float elapsedTime;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start, 0);
	
    int out=1;
    int iteration = 0;
    do{    
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
        
        call_solver(ni, nj, n, f, fm1, fm2, dt_host, pre, gridDim, blockDim);
        
        cudaDeviceSynchronize();
        
        
        time0 = time0 + dt_host;
        
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
                
                // rank 0 copies data to host
                cudaMemcpy(h_x, x+1, ni * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_y, y, nj * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_f, f + 1 * m, n * m * sizeof(double), cudaMemcpyDeviceToHost);
                // receive data from other ranks
                if (size > 1) {
                    MPI_Request request[size-1];
                    for(int ran = 1;ran < size;ran++){
                        int i = ni / size; // 获取当前 rank 的数据行数
                        int j = nj;
                        if(ran==size-1){
                            i +=  ni % size;
                        }  
                        MPI_Irecv(h_f+ (ran * ni / size) * j, i*j , MPI_DOUBLE, ran , 0, MPI_COMM_WORLD, request + ran - 1);
                    }
                    MPI_Waitall(size-1, request, MPI_STATUSES_IGNORE);
                }
            }
            else {
                MPI_Request request;
                // ! may be wrong
                // 给 h_f 分配内存
                h_f = new double[n * m];
                // 拷贝数据到 host
                cudaMemcpy(h_f, f + 1 * m, n * m * sizeof(double), cudaMemcpyDeviceToHost);
                // 发送数据到 rank 0
                MPI_Isend(h_f, n*m , MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                delete[] h_f;
            }
            
            if(rank==0){
                call_output(output_count , rank, ni, nj, h_x, h_y, h_f);	
            }
        } 
        
        printf("Output written for time step %d\n", iteration);
    } while (time0 <= tout);
	
	// 记录结束时间
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	// 计算经过的时间
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("====================================================\n");
    printf("rank = %d, n = %d, m = %d\n", rank, n, m);
    printf("Total time: %.4f s\n", elapsedTime/1000.0);
    printf("Average time per time step (GPU): %.4f s\n", elapsedTime / iteration / 1000.0);
    
    if(rank==0){
        printf("Total output files: %d\n", output_count);
    }

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    call_releasespace();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize(); 
    return 0;
}


NVCC = nvcc
CXX = mpicxx

CUDA_PATH = /usr/local/cuda

MPI_CFLAGS = $(shell mpicxx --showme:compile)

NVCC_FLAGS += -ccbin /usr/bin/gcc-11 $(MPI_CFLAGS)

CXX_FLAGS = -O3 -I. -I$(CUDA_PATH)/include

CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart

TARGET = main
CU_SOURCES = kernel.cu
CPP_SOURCES = main.cpp
OBJS = kernel.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(CUDA_LIBS)

kernel.o: $(CU_SOURCES) Jacobi.h
	$(NVCC) $(NVCC_FLAGS) -c $(CU_SOURCES) -o kernel.o

main.o: $(CPP_SOURCES) Jacobi.h
	$(CXX) $(CXX_FLAGS) -c $(CPP_SOURCES) -o main.o

clean:
	rm -f $(OBJS) 
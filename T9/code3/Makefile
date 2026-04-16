# Makefile for CUDA + MPI project

# 编译器
NVCC := nvcc -ccbin g++-11

# MPI 路径配置 (根据 mpicc -show 的结果设置)
MPI_HOME := /usr/local/openmpi-4.1.5-cuda
MPI_INC := -I$(MPI_HOME)/include
MPI_LIB := -L$(MPI_HOME)/lib -lmpi

# 目录路径
INCLUDE_DIR := include
SRC_DIR := src
OBJ_DIR := obj

# 目标文件
TARGET := main

# 获取所有 .cu 源文件
SRCS := $(wildcard $(SRC_DIR)/*.cu)
# 生成对应的 .o 文件路径
OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

# 编译选项
NVCC_FLAGS := -I$(INCLUDE_DIR) $(MPI_INC) -O3 -std=c++11 -rdc=true

# 链接选项
LDFLAGS := $(MPI_LIB) -Xlinker -rpath -Xlinker $(MPI_HOME)/lib -rdc=true

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)
# 	@echo "Build successful! Executable created at: $@"

# 编译源文件生成目标文件
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 创建对象文件目录
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# 清理规则
clean:
	rm -rf $(OBJ_DIR) 

.PHONY: all clean

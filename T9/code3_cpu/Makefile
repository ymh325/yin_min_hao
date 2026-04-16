# Makefile for CPU + MPI project

# 编译器
CXX := mpicxx

# 目录路径
INCLUDE_DIR := include
SRC_DIR := src
OBJ_DIR := obj

# 目标文件
TARGET := main

# 获取所有 .cpp 源文件
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# 生成对应的 .o 文件路径
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# 编译选项
CXXFLAGS := -I$(INCLUDE_DIR) -O3 -std=c++11

# 链接选项
LDFLAGS :=

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# 编译源文件生成目标文件
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 创建对象文件目录
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# 清理规则
clean:
	rm -rf $(OBJ_DIR)

.PHONY: all clean

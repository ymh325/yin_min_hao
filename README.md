# 并行计算实验合集



> [!done] 说明 
> 本仓库主要整理了 MPI、CUDA 和并行数值计算相关的课程实验与研究代码，内容覆盖 Jacobi 迭代、矩阵乘法优化、RMA 通信实验以及一个 CPU 版并行求解器示例。


## 目录说明


### T8
- `T8/Jacobi/`：CUDA + MPI 的 Jacobi 迭代实现，按进程划分网格并在 GPU 上计算。
- `T8/one/`：CUDA 示例程序，包含 `jacobi.cu` 和 `te.cu`。
- `T8/矩阵乘法优化/`：矩阵乘法优化实验，包含多个版本的实现。

### T9
- `T9/code3_cpu/`：CPU 版 MPI 并行求解程序，按进程划分二维网格并进行迭代计算。
- `T9/code3/`：另一份 `code3` 实验代码，结构与 `code3_cpu/` 类似。

### research

- `research/RMA_BASE/`：MPI RMA 基础实验，包含 `block`、`non-block`、`Lock`、`Share`、`PUT_GET+PSCW` 等同步模式的实现。

- `research/RMA_deep/`：关于代码复现，主要是对于RMA在CFD领域应用的文献

## 环境要求

- Linux
- MPI 编译器：`mpicxx` / `mpiexec`
- CUDA 工具链：用于 `T8/Jacobi/` 和 `T8/one/` 中的 GPU 程序
- 常见编译工具：`make`、`gcc`/`g++`

## 典型构建方式

不同子目录的构建方式不完全一致，通常在对应目录下执行：

```bash
make
```

其中：

- `T8/Jacobi/` 通过 `nvcc` 编译 `.cu` 文件，并使用 `mpicxx` 链接。
- `T9/code3_cpu/` 使用 `mpicxx` 编译 `src/` 下的源码。
- `research/RMA_BASE/` 和 `research/RMA_deep/2014/` 通过各自的 `makefile` 生成可执行文件，随后用 `mpiexec` 运行。

## 运行说明

多数 MPI 程序都需要通过进程数启动，例如：

```bash
mpiexec -n 4 ./main
```

如果程序涉及 GPU，需确保每个 MPI 进程可访问可用的 CUDA 设备。


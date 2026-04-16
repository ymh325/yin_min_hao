# PUT/GET + PSCW

`RMA` 中的一种同步方式。

```cpp

MPI_Win_post 
MPI_Win_start
MPI_Win_complete
MPI_Win_wait
```

简称 `PSCW`

特点：
- 双方提前约定通信
- 类似 **同步窗口**

# PUT/GET + lock

另一种 `RMA` 的同步方式。

函数：
```cpp
MPI_Win_lock
MPI_Win_unlock
```

特点：
- 单边访问
- `target` 进程不用参与

EG：
```cpp
MPI_Win_lock(...)
MPI_Put(...)
MPI_Win_unlock(...)
```

# Shared Memory

这是 `MPI` 的共享内存模式

函数：
```cpp
MPI_Win_allocate_shared
MPI_Win_shared_query
```

特点：
- 如果多个进程在 **同一个结点：** 多个进程直接访问一块内存，不需要网络通信
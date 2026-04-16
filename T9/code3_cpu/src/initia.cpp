#include <iostream>
#include "Module.h"

void call_initia(int rank, int n, int m, double* f, double* fm1, double* fm2, double* x, double* y, int pre)
{
    for (int local_i = 1; local_i <= n; local_i++) {
        int global_i = local_i + pre;

        // 边界检查
        if (global_i > ni) {
            continue;
        }

        for (int j = 0; j < nj; j++) {
            int idx = local_i * m + j;

            f[idx] = 0.0;
            fm1[idx] = 0.0;
            fm2[idx] = 0.0;

            // 设置初始条件（用 global_i 查全局坐标数组）
            if (0.2 <= x[global_i] && x[global_i] <= 0.5 && 0.2 <= y[j] && y[j] <= 0.5)
            {
                f[idx] = 1;
            }
        }
    }
    return;
}

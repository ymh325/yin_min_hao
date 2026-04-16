//---------------------------------------------------------------------------//
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include "Module.h"
using namespace std;

void call_output(double num, int rank, int ni, int nj, double* x, double* y, double* f)
{
    // 创建输出文件（二进制模式）
    std::ostringstream fname_ss;
    fname_ss << "output/file" << static_cast<int>(num) << ".bin";
    std::string filename = fname_ss.str();
    FILE* fp = fopen(filename.c_str(), "wb");

    if (!fp) {
        printf("ERROR: Failed to open file %s\n", filename.c_str());
        exit(1);
    }
    printf("Writing output file: %s\n", filename.c_str());

    // 写入网格维度信息（二进制头）
    fwrite(&ni, sizeof(int), 1, fp);
    fwrite(&nj, sizeof(int), 1, fp);

    // 写入 x 坐标 (ni 个 double)
    fwrite(x, sizeof(double), ni, fp);

    // 写入 y 坐标 (nj 个 double)
    fwrite(y, sizeof(double), nj, fp);

    // 写入 f 数据 (ni*nj 个 double)
    fwrite(f, sizeof(double), ni * nj, fp);

    fclose(fp);

    delete[] x;
    delete[] y;
    delete[] f;
    // 释放主机内存
    return;
}

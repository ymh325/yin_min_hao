//---------------------------------------------------------------------------//
#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<string>
#include<cstdio>
#include"Module.cuh"
using namespace std;
void call_output(double num, int rank,int ni, int nj, double* x, double* y, double* f)
{	
    std::ostringstream fname_ss;
	fname_ss << "output/file" << static_cast<int>(num) << ".dat";
	std::string filename = fname_ss.str();
	ofstream outfile(filename.c_str());
	
	// 写入文件头
	outfile << "VARIABLEs = X,Y,fi" << '\n';
	outfile << "ZONE I=" << ni << '\t' << "J=" << nj - 1 << '\n';
	outfile << "datapacking=block" << '\n';

   for (int j = 0; j < nj; j++)
		for (int i = 0; i < ni; i++)
			outfile << x[i] << '\n';
	
	for (int j = 0; j < nj; j++)
		for (int i = 0; i < ni; i++)
			outfile << y[j] << '\n';
	
	for (int j = 0; j < nj; j++)	
		for (int i = 0; i < ni; i++)
			outfile << f[i * nj + j] << '\n';
    
    outfile.close();

    delete[] x;
    delete[] y;
    delete[] f;
    // 释放主机内存
    return;
}
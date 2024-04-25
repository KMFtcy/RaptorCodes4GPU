#include <iostream>
#include "main.cuh"

using namespace std;

int main(){
    cout<< "Hello, R11 for GPU!" << endl;

    const int N = 1000;
    size_t BytesCount = sizeof(float)*N;

    //allocate memory for data
    float *d_x;
    cudaMalloc((float **)&d_x,BytesCount);
    cudaMemset(d_x, 0, BytesCount); 
    cout << "Allocate " << BytesCount << "Bytes" << endl;
    float *d_y;
    cudaMalloc((float **)&d_y,BytesCount);
    cudaMemset(d_y, 1, BytesCount);
    // first 10 of y
    showFirstNonGPU(d_y, 10);

    cudaRaptorParam params;
    unsigned int K = 44;
    params.K = K;
    params.Kmin = 1024; //a minimum target on the number of symbols per source block
    params.Kmax = 8192; //the maximum number of source symbols per source block.
    params.Gmax = 10; //a maximum target number of symbols per packet
    params.Al = 4; //the symbol alignment parameter, in bytes, 一个symbol的长度
    params.N = 24;
    params.T = 4; // symbol size, in bytes

    cudaR10_compute_params(&params);
    // printCudaRaptorParam(params);

    // 5、调用核函数在设备中进行计算
    dim3 block(1);
    dim3 grid(10);

    cudaLTEnc<<<grid, block>>>(d_x,d_y, N);
    cudaDeviceSynchronize(); //cudaMemcpy有隐式的同步
    showFirstNonGPU(d_y, 10);

    //free memory
    cudaFree(d_x);
    cudaFree(d_y);
}
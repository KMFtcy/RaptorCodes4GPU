#include <iostream>
#include "main.cuh"
#include <stdint.h>
#include "raptor_consts.h"

using namespace std;

int main(){
    cout<< "Hello, R11 for GPU!" << endl;

    const int K = 1000;
    size_t BytesCount = sizeof(float)*K;

    //allocate memory for data
    word *d_x;
    cudaMalloc((word **)&d_x,BytesCount);
    cudaMemset(d_x, 0, BytesCount); 
    cout << "Allocate " << BytesCount << "Bytes" << endl;
    word *d_y;
    cudaMalloc((word **)&d_y,BytesCount);
    cudaMemset(d_y, 1, BytesCount);

    cudaRaptorParam params;
    // unsigned int K = 44;
    params.K = K;
    params.Kmin = 1024; //a minimum target on the number of symbols per source block
    params.Kmax = 8192; //the maximum number of source symbols per source block.
    params.Gmax = 10; //a maximum target number of symbols per packet
    params.Al = 4; //the symbol alignment parameter, in bytes, 一个symbol的长度
    params.N = 24;
    params.T = 4; // symbol size, in bytes
    cudaR10_compute_params(&params);

    // Copy ramdom table from host to device
    uint32_t *device_J;
    uint32_t *device_V0;
    uint32_t *device_V1;
    cudaMalloc((uint32_t **)&device_J,10901);
    cudaMalloc((uint32_t **)&device_V0,250);
    cudaMalloc((uint32_t **)&device_V1,250);
    cudaMemcpy(device_J, J, 10901, cudaMemcpyHostToDevice);
    cudaMemcpy(device_V0, V0, 250, cudaMemcpyHostToDevice);
    cudaMemcpy(device_V1, V1, 250, cudaMemcpyHostToDevice);

    // Call kernel function to compute on GPU
    dim3 block(1);
    dim3 grid(10);

    cudaLTEnc<<<grid, block>>>(K, d_x, d_y, params.L, 24, device_J, device_V0, device_V1);
    cudaDeviceSynchronize();
    showFirstNonGPU(d_y,10);

    //free memory
    cudaFree(d_x);
    cudaFree(d_y);
}
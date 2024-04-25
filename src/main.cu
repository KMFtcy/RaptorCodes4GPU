#include <iostream>
#include "main.cuh"

using namespace std;

int main(){
    cout<< "Hello, R11 for GPU!" << endl;

    const int N = 100000000;
    const int M = sizeof(float)*N;

    //allocate memory for data
    float *d_x;
    cudaMalloc((void **)&d_x,M);
    cout << "Allocate " << M << "Bytes" << endl;

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
    printCudaRaptorParam(params);

    //free memory
    cudaFree(d_x);
}
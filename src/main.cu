#include <iostream>
#include <main.hpp>

using namespace std;

int main(){
    cout<< "Hello, R11 for GPU!" << endl;

    const int N = 100000000;
    const int M = sizeof(float)*N;

    //allocate memory for data
    float *d_x;
    cudaMalloc((void **)&d_x,M);
    cout << "Allocate " << M << "Bytes" << endl;

    //free memory
    cudaFree(d_x);
}
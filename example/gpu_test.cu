#include <iostream>
#include <stdint.h>
#include "cuda_raptor_10.cuh"
// #include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include "raptor10.hpp"

using namespace std;

vector<char>* multiply_symbols(vector<char>& A, vector<char>& B) {
    if (A.size() != B.size()) {
        return NULL;
    }

    int size = A.size();
    vector<char>* C = new vector<char>(size, 0);
    for (int i = 0; i < size; i++) {
        (*C)[i] = A[i] ^ B[i];
    }
    return C;
}

__global__ void d_multiply_symbols(char* A, char* B, char* C){
    int threadIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

    *(C + threadIdx.x) = *(A + threadIdx.x) ^ *(B + threadIdx.x);
}

int main()
{
    cout << "GPU test begin" << endl;

    // Initialize paramaters
    Raptor10 params;
    params.Kmin = 1024; // a minimum target on the number of symbols per source block
    params.Kmax = 8192; // the maximum number of source symbols per source block.
    params.Gmax = 10;   // a maximum target number of symbols per packet
    params.T = 1024;    // symbol size, suppose to be a ip packet size
    params.K = 4000;
    params.Al = 4; // the symbol alignment parameter, in bytes, 一个symbol的长度
    int L_ = params.L;
    r10_compute_params(&params);
    params.N = params.K + params.S + 10;
    cout << "K = " << params.K;
    cout << ", S = " << params.S;
    cout << ", H = " << params.H;
    cout << ", L = " << params.L ;
    cout << ", N = " << params.N << endl;
    // LT coding params
    while (!is_prime(L_))
        L_++;
    vector<int> ESIs(params.N); // Create vector of ESIs
    for (int i = 0; i < params.N; i++){
        ESIs[i] = i;
    }


    cout << "test2" << endl;
    // prepare container for data and encoded data
    char data[params.K][params.T];
    char encoded_data[params.K][params.T];

    for (int i = 0; i < params.K; i++)
    {
        for (int j = 0; j < params.T; j++)
        {
            data[i][j] = rand() % 256;
            encoded_data[i][j] = data[i][j];
        }
    }

    cout << "Data: " << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << (int)data[i][0] << " | ";
    }
    cout << endl;

    // allocate device memory
    char **d_data, **d_encoded_data;
    int data_size = params.K * params.T * sizeof(char);
    int encoded_data_size = params.N * params.T * sizeof(char);

    cudaMalloc(d_data, data_size);
    cudaMalloc(d_encoded_data, encoded_data_size);

     // copy data to device memory
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_encoded_data, encoded_data, encoded_data_size, cudaMemcpyHostToDevice);

    // start coding on GPU
    int threadsPerBlock = params.T;
    int blocksPerGrid = params.K;
    // start LDPC
    clock_t start, end;
    int a = 0, b = 0;
    start = clock();
    for (int i = 0; i < params.K; i++)
    {
        a = 1 + ((int)floor(i / params.S) % (params.S - 1));
        b = i % params.S;
        d_multiply_symbols<<<1, threadsPerBlock>>>(d_encoded_data[i], d_encoded_data[params.K + b], d_encoded_data[params.K + b]);
        b = (b + a) % params.S;
        d_multiply_symbols<<<1, threadsPerBlock>>>(d_encoded_data[i], d_encoded_data[params.K + b], d_encoded_data[params.K + b]);
        b = (b + a) % params.S;
        d_multiply_symbols<<<1, threadsPerBlock>>>(d_encoded_data[i], d_encoded_data[params.K + b], d_encoded_data[params.K + b]);
        cudaDeviceSynchronize();
    }

    cout << "test" << endl;
    // start LDPC
    // LT coding
    for (uint32_t i = params.L; i < params.N; i++) {
        uint32_t triple[3] = {0};
        uint32_t X = ESIs[i];
        r10_Trip(params.K, X, triple, &params);
        uint32_t d = triple[0];
        uint32_t a = triple[1];
        uint32_t b = triple[2];
        uint32_t j_max = fmin((d - 1), (params.L - 1));

        while (b >= params.L){
          b = (b + a) % L_;
        }

        d_encoded_data[i] = d_encoded_data[b];

        for (int j = 1; j <= j_max; j++) {
          b = (b + a) % L_;

          while (b >= params.L)
              b = (b + a) % L_;

            d_multiply_symbols<<<1, threadsPerBlock>>>(d_encoded_data[i], d_encoded_data[b], d_encoded_data[i]);
        }
        cudaDeviceSynchronize();
    }

    end = clock();

    // copy data from device memory to host
    // cudaMemcpy(data_flat.data(), d_data, data_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(encoded_data_flat.data(), d_encoded_data, encoded_data_size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_data);
    cudaFree(d_encoded_data);

    // analysis
    double running_time = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Coded data size: " << data_size/1000 << "kbyte" << endl;
    cout << "Coding time: " << running_time << "s" << endl;
    cout << "Coding rate: " << static_cast<int>(data_size / (1000000 * running_time)) << "MB/s" << endl;

}
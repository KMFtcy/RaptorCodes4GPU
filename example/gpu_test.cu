#include <iostream>
#include <stdint.h>
#include "cuda_raptor_10.cuh"
// #include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include "raptor10.hpp"

using namespace std;

vector<char> *multiply_symbols(vector<char> &A, vector<char> &B)
{
    if (A.size() != B.size())
    {
        return NULL;
    }

    int size = A.size();
    vector<char> *C = new vector<char>(size, 0);
    for (int i = 0; i < size; i++)
    {
        (*C)[i] = A[i] ^ B[i];
    }
    return C;
}

__global__ void d_multiply_symbols(char *A, char *B, char *C)
{
    int threadIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

    *(C + threadIdx.x) = *(A + threadIdx.x) ^ *(B + threadIdx.x);
}

void encoding(Raptor10 &param, char **A, char **data_dev, char **encoded_data_dev)
{
    // Generate intermediate symbols
    char **D;
    cudaMalloc(&D, param.L * sizeof(char *));
    for (int i = 0; i < param.L; i++)
    {
        char *d_row;
        cudaMalloc(&d_row, param.T * sizeof(char));
        cudaMemcpy(&D[i], &d_row, sizeof(char *), cudaMemcpyHostToDevice);
    }
    init_D<<<1, 1>>>(param.L, param.K, param.T, D, data_dev);
    cudaDeviceSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    gaussianElimination<<<1, 1>>>(A, D, param.L, param.L, param.T); // now D is the intermediate symbols
    cudaDeviceSynchronize();

    // LT coding
    cudaMemcpy(encoded_data_dev, data_dev, param.K * sizeof(char *), cudaMemcpyDeviceToDevice);
    vector<int> ESIs_h(param.N - param.K); // Create vector of ESIs of sending symbols
    for (int i = 0; i < param.N - param.K; i++)
    {
        ESIs_h[i] = param.K + i;
    }
    int* ESIs_d;
    cudaMalloc(&ESIs_d, ESIs_h.size() * sizeof(int));
    cudaMemcpy(ESIs_d, ESIs_h.data(), ESIs_h.size() * sizeof(int), cudaMemcpyHostToDevice);
    // create ramdom table
    uint32_t *d_J;
    uint32_t *d_V0;
    uint32_t *d_V1;
    const size_t J_size = sizeof(J);
    const size_t V0_size = sizeof(V0);
    const size_t V1_size = sizeof(V1);
    cudaMalloc((void **)&d_J, J_size);
    cudaMalloc((void **)&d_V0, V0_size);
    cudaMalloc((void **)&d_V1, V1_size);
    cudaMemcpy(d_J, J, J_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V0, V0, V0_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V1, V1, V1_size, cudaMemcpyHostToDevice);
    // printRandomTable<<<1, 1>>>(d_J, d_V0, d_V1);
    cout << "Start LT coding" << endl;
    LTEnc<<<1, 1>>>(
        param.K, param.S, param.H, param.T, param.LP, param.N - param.K, ESIs_d, D, encoded_data_dev, d_J, d_V0, d_V1);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

void decoding(Raptor10 &params, char **data_dev, char **encoded_data_dev)
{
    int L_ = params.L;
    while (!is_prime(L_))
        L_++;
    vector<int> ESIs(params.N); // Create vector of ESIs of sending symbols
    for (int i = 0; i < params.N; i++)
    {
        ESIs[i] = i;
    }
}

int main()
{
    cout << "GPU test begin" << endl;

    // Initialize paramaters
    Raptor10 param;
    param.Kmin = 1024; // a minimum target on the number of symbols per source block
    param.Kmax = 8192; // the maximum number of source symbols per source block.
    param.Gmax = 10;   // a maximum target number of symbols per packet
    param.T = 15;    // symbol size
    param.K = 10;
    int overhead = 5;
    r10_compute_params(&param, overhead);
    cout << "K = " << param.K;
    cout << ", S = " << param.S;
    cout << ", H = " << param.H;
    cout << ", L = " << param.L;
    cout << ", N = " << param.N << endl;

    // Allocate test data
    // create ESIs
    vector<int> ESIs(param.N);
    for (int i = 0; i < param.N; i++){
        ESIs[i] = i;
    }
    int* ESIs_d;
    cudaMalloc(&ESIs_d, ESIs.size() * sizeof(int));
    cudaMemcpy(ESIs_d, ESIs.data(), ESIs.size() * sizeof(int), cudaMemcpyHostToDevice);

    // prepare data
    char **data;
    data = (char **)malloc(param.K * sizeof(char *));
    for (int i = 0; i < param.K; i++)
    {
        data[i] = (char *)malloc(param.T * sizeof(char));
    }
    for (int i = 0; i < param.K; i++)
    {
        for (int j = 0; j < param.T; j++)
        {
            data[i][j] = rand() % 256;
        }
    }

    char **encoded_data;
    encoded_data = (char **)malloc(param.N * sizeof(char *));
    for (int i = 0; i < param.N; i++)
    {
        encoded_data[i] = (char *)malloc(param.T * sizeof(char));
    }

    // copy data to device memory
    char **data_dev, **encoded_data_dev;
    int data_size = param.K * param.T * sizeof(char);
    int encoded_data_size = param.N * param.T * sizeof(char);
    cout << "data size: " << data_size << " bytes" << endl;

    // Allocate memory on the device
    cudaMalloc(&data_dev, param.K * sizeof(char *));
    cudaMalloc(&encoded_data_dev, param.N * sizeof(char *));

    // Allocate memory for each row on the device
    for (int i = 0; i < param.K; ++i)
    {
        char *d_row;
        cudaMalloc(&d_row, param.T * sizeof(char));
        cudaMemcpy(d_row, data[i], param.T * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(&data_dev[i], &d_row, sizeof(char *), cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < param.N; ++i)
    {
        char *d_row;
        cudaMalloc(&d_row, param.T * sizeof(char));
        cudaMemcpy(d_row, encoded_data[i], param.T * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(&encoded_data_dev[i], &d_row, sizeof(char *), cudaMemcpyHostToDevice);
    }

    // Time recording
    clock_t start, end;

    // Start coding on GPU
    // Generate A matrix
    char **A = Matrix_A_Generator(param, ESIs_d, param.K);
    cout << "Matrix A Generated" << endl;
    // print_matrix(param.L, param.L, A);
    start = clock();
    encoding(param, A, data_dev, encoded_data_dev);
    end = clock();
    double encoding_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Randomly drop some symbols
    random_loss(ESIs_d, encoded_data_dev);

    // Decoding on GPU

    // Check if the result is correct

    // Free device memory

    // Analysis
    double running_time = encoding_time;
    cout << "Coded data size: " << data_size / 1000.0 << "kbyte" << endl;
    cout << "Coding time: " << running_time << "s" << endl;
    cout << "Coding rate: " << static_cast<double>(data_size * 1.0 / (1000000 * running_time)) << "MB/s" << endl;
}
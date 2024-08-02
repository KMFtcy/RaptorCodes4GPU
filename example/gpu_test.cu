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

void encoding(Raptor10 &param, char **data_dev, char **encoded_data_dev)
{
    // Generate intermediate symbols
    // Generate A matrix
    std::vector<std::vector<char>> _A(param.L, std::vector<char>(param.L, 0));

    // Allocate device pointer array
    char **A;
    cudaMalloc(&A, param.L * sizeof(char *));

    // Allocate device memory for each row and copy data
    for (int i = 0; i < param.L; ++i)
    {
        char *d_row;
        cudaMalloc(&d_row, param.L * sizeof(char));
        cudaMemcpy(d_row, _A[i].data(), param.L * sizeof(char), cudaMemcpyHostToDevice);
        // Copy device row pointer to device pointer array
        cudaMemcpy(&A[i], &d_row, sizeof(char *), cudaMemcpyHostToDevice);
    }

    vector<int> h_ESIs(param.K); // Create vector of ESIs of sending symbols
    for (int i = 0; i < param.K; i++)
    {
        h_ESIs[i] = i;
    }
    int *ESIs;
    cudaMalloc(&ESIs, h_ESIs.size() * sizeof(int));
    cudaMemcpy(ESIs, h_ESIs.data(), h_ESIs.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Create ramdom table
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

    cudaError_t error;
    LDPC_Matrix_Generator<<<1, 1>>>(param.K, param.S, A);
    HALF_Matrix_Generator<<<1, 1>>>(param.K, param.S, param.H, param.HP, A);
    I_S_Matrix_Generator<<<1, 1>>>(param.K, param.S, A);
    I_H_Matrix_Generator<<<1, 1>>>(param.K, param.S, param.H, A);
    G_LT_Matrix_Generator<<<1, 1>>>(param.K, param.S, param.H, param.L, param.LP, A, ESIs, param.K, d_J, d_V0, d_V1);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    cout << "Matrix A Generated" << endl;
    print_matrix_A(param, A);
    // generate intermediate symbols

    // LT coding
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
    param.T = 1500;    // symbol size
    param.K = 10;
    int overhead = 5;
    r10_compute_params(&param, overhead);
    cout << "K = " << param.K;
    cout << ", S = " << param.S;
    cout << ", H = " << param.H;
    cout << ", L = " << param.L;
    cout << ", N = " << param.N << endl;

    // Allocate test data
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
    start = clock();
    encoding(param, data_dev, encoded_data_dev);
    end = clock();
    double encoding_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Randomly drop some symbols

    // Decoding on GPU

    // Check if the result is correct

    // Free device memory

    // Analysis
    double running_time = encoding_time;
    cout << "Coded data size: " << data_size / 1000 << "kbyte" << endl;
    cout << "Coding time: " << running_time << "s" << endl;
    cout << "Coding rate: " << static_cast<int>(data_size / (1000000 * running_time)) << "MB/s" << endl;
}
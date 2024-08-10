#include <iostream>
#include <stdint.h>
#include "cuda_raptor_10.cuh"
#include <vector>
#include <numeric>
#include <math.h>
#include "raptor10.hpp"

using namespace std;

const int packet_size = 1500;

void encoding(Raptor10 &param, process_unit **A, process_unit **source_symbols_d, process_unit **encoded_symbols_d)
{
    clock_t start, end;
    // Generate intermediate symbols
    process_unit **D;
    cudaMalloc(&D, param.L * sizeof(process_unit *));
    for (int i = 0; i < param.L; i++)
    {
        process_unit *d_row;
        cudaMalloc(&d_row, param.T * sizeof(process_unit));
        cudaMemcpy(&D[i], &d_row, sizeof(process_unit *), cudaMemcpyHostToDevice);
    }
    init_D<<<packet_size, 1>>>(param.L, param.K, param.T, D, source_symbols_d);
    cudaDeviceSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    start = clock();
    gaussianElimination(A, D, param.L, param.L, param.T, packet_size); // now D is the intermediate symbols
    cudaDeviceSynchronize();
    end = clock();
    printf("gaussianElimination time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    // LT coding
    cudaMemcpy(encoded_symbols_d, source_symbols_d, param.K * sizeof(process_unit *), cudaMemcpyDeviceToDevice);
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
    LTEnc<<<packet_size, 1>>>(
        param.K, param.S, param.H, param.T, param.LP, param.N - param.K, ESIs_d, D, encoded_symbols_d, d_J, d_V0, d_V1);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

void decoding(Raptor10 &param,process_unit** A, process_unit **decoded_symbols_d,process_unit **received_symbols_d,  int *ESIs, int N)
{
    clock_t start, end;
    int M = N + param.S + param.H;
    // Generate intermediate symbols
    process_unit **D;
    cudaMalloc(&D, M * sizeof(process_unit *));
    for (int i = 0; i < M; i++)
    {
        process_unit *d_row;
        cudaMalloc(&d_row, param.T * sizeof(process_unit));
        cudaMemcpy(&D[i], &d_row, sizeof(process_unit *), cudaMemcpyHostToDevice);
    }
    init_D<<<packet_size, 1>>>(M, N, param.T, D, received_symbols_d);
    cudaDeviceSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    start = clock();
    gaussianElimination(A, D, M, param.L, param.T, packet_size); // now the previous L symbols of D is the intermediate symbols
    cudaDeviceSynchronize();
    end = clock();
    printf("gaussianElimination time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Generate the missing ESI array
    // int *all_ESIs = (int *)malloc(param.K * sizeof(int));
    // for (int i = 0; i < param.K; i++)
    // {
    //     all_ESIs[i] = i;
    // }
    // int* ESIs_h = (int *)malloc(N * sizeof(int));;
    // cudaMemcpy(ESIs_h, ESIs, N * sizeof(int), cudaMemcpyDeviceToHost);
    // int *received_ESIs = (int *)malloc(N * sizeof(int));
    // for (int i = 0; i < N; i++)
    // {
    //     received_ESIs[i] = ESIs_h[i];
    // }

    // int *missing_ESIs = (int *)malloc((param.K - N) * sizeof(int));
    // int missing_count = 0;
    // for (int i = 0; i < param.K; i++)
    // {
    //     int found = 0;
    //     for (int j = 0; j < N; j++)
    //     {
    //         if (all_ESIs[i] == received_ESIs[j])
    //         {
    //             found = 1;
    //             break;
    //         }
    //     }
    //     if (!found)
    //     {
    //         missing_ESIs[missing_count++] = all_ESIs[i];
    //     }
    // }
    // int *missing_ESIs_d;
    // cudaMalloc(&missing_ESIs_d, missing_count * sizeof(int));
    // cudaMemcpy(missing_ESIs_d, missing_ESIs, missing_count * sizeof(int), cudaMemcpyHostToDevice);

    // LT coding to recover lost symbols
    vector<int> ESIs_h(param.K); // Create vector of ESIs of sending symbols
    for (int i = 0; i < param.K; i++)
    {
        ESIs_h[i] = i;
    }
    int* ESIs_d;
    cudaMalloc(&ESIs_d, ESIs_h.size() * sizeof(int));
    cudaMemcpy(ESIs_d, ESIs_h.data(), ESIs_h.size() * sizeof(int), cudaMemcpyHostToDevice);
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

    LTEnc<<<param.T, 1>>>(param.K, param.S, param.H, param.T, param.LP, param.K, ESIs_d, D, decoded_symbols_d, d_J, d_V0, d_V1);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
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
    param.T = packet_size;    // symbol size
    param.K = 100;
    int overhead = 10;
    int loss = 3;
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
    process_unit **data;
    data = (process_unit **)malloc(param.K * sizeof(process_unit *));
    for (int i = 0; i < param.K; i++)
    {
        data[i] = (process_unit *)malloc(param.T * sizeof(process_unit));
    }
    for (int i = 0; i < param.K; i++)
    {
        for (int j = 0; j < param.T; j++)
        {
            data[i][j] = rand() % 128;
        }
    }

    process_unit **encoded_data;
    encoded_data = (process_unit **)malloc(param.N * sizeof(process_unit *));
    for (int i = 0; i < param.N; i++)
    {
        encoded_data[i] = (process_unit *)malloc(param.T * sizeof(process_unit));
        memset(encoded_data[i], 0, param.T * sizeof(process_unit));
    }

    // copy data to device memory
    process_unit **source_symbols_d, **encoded_symbols_d;
    int data_size = param.K * param.T * sizeof(process_unit);
    int encoded_data_size = param.N * param.T * sizeof(process_unit);
    cout << "data size: " << data_size << " bytes" << endl;

    // Allocate memory on the device
    cudaMalloc(&source_symbols_d, param.K * sizeof(process_unit *));
    cudaMalloc(&encoded_symbols_d, param.N * sizeof(process_unit *));

    // Allocate memory for each row on the device
    for (int i = 0; i < param.K; ++i)
    {
        process_unit *d_row;
        cudaMalloc(&d_row, param.T * sizeof(process_unit));
        cudaMemcpy(d_row, data[i], param.T * sizeof(process_unit), cudaMemcpyHostToDevice);
        cudaMemcpy(&source_symbols_d[i], &d_row, sizeof(process_unit *), cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < param.N; ++i)
    {
        process_unit *d_row;
        cudaMalloc(&d_row, param.T * sizeof(process_unit));
        cudaMemcpy(d_row, encoded_data[i], param.T * sizeof(process_unit), cudaMemcpyHostToDevice);
        cudaMemcpy(&encoded_symbols_d[i], &d_row, sizeof(process_unit *), cudaMemcpyHostToDevice);
    }

    // Time recording
    clock_t start, end;

    // Start coding on GPU
    // Generate A matrix
    process_unit **A = Matrix_A_Generator(param, ESIs_d, param.K);
    cout << "Matrix A Generated" << endl;
    start = clock();
    encoding(param, A, source_symbols_d, encoded_symbols_d);
    end = clock();
    double encoding_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Randomly drop some symbols
    cout << "Randomly drop some symbols" << endl;
    int S = param.N - loss;
    int* remained_ESIs_d = ESIs_d + loss;
    process_unit **remained_encoded_symbols_d = encoded_symbols_d + loss;

    // Decoding on GPU
    cout << "Decoding on GPU" << endl;
    process_unit **decode_A = Matrix_A_Generator(param, remained_ESIs_d, param.N - loss);
    process_unit** decoded_data_d;
    cudaMalloc(&decoded_data_d, param.K * sizeof(process_unit *));
    for (int i = 0; i < param.K; ++i)
    {
        process_unit *d_row;
        cudaMalloc(&d_row, param.T * sizeof(process_unit));
        cudaMemcpy(&decoded_data_d[i], &d_row, sizeof(process_unit *), cudaMemcpyHostToDevice);
    }
    start = clock();
    decoding(param, decode_A, decoded_data_d, remained_encoded_symbols_d, remained_ESIs_d, param.N - loss);
    end = clock();
    double decoding_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Check if the result is correct
    // cout << "source symbols:" << endl;
    // print_matrix(param.K, param.T, source_symbols_d);
    // cout << "decoded symbols:" << endl;
    // print_matrix(param.K, param.T, decoded_data_d);
    // cout << "encoded symbols:" << endl;
    // print_matrix(param.N, param.T, encoded_symbols_d);
    // cout << "received symbols:" << endl;
    // print_matrix(param.N - 3, param.T, remained_encoded_symbols_d);
    check_result<<<1, 1>>>(source_symbols_d, decoded_data_d, param.K, param.T);
    cudaError_t error;
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    } else {
        cout << "The result is correct" << endl;
    }

    // Free device memory

    // Analysis
    cout << "Successfully decoded!" << endl;
    cout << "Coded data size: " << data_size / 1000.0 << "kbyte" << endl;
    cout << "Encoding time: " << encoding_time << "s" << endl;
    cout << "Encoding rate: " << static_cast<double>(data_size * 4.0 / (1000000 * encoding_time)) << "MB/s" << endl;
    cout << "Decoding time: " << decoding_time << "s" << endl;
    cout << "Decoding rate: " << static_cast<double>(data_size * 4.0 / (1000000 * decoding_time)) << "MB/s" << endl;
}
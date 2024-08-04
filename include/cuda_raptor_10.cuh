#ifndef CUDA_RAPTOR_10_CUH
#define CUDA_RAPTOR_10_CUH

#include <stdint.h>
#include <cuda_raptor_10.h>
#include <raptor10.hpp>
#include <vector>

__global__ void cudaLTEncImpl(const int K, word *C, word *EncC, const int L, const int N, uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1);

__device__ void processData(char *d_data, int data_flat_size);

__global__ void test_Modify_A(int L, char **A);

__global__ void printRandomTable(const uint32_t *d_J, const uint32_t *d_V0, const uint32_t *d_V1);

__global__ void LDPC_Matrix_Generator(int K, int S, char **A);

__global__ void HALF_Matrix_Generator(int K, int S, int H, int HP, char **A);

__global__ void I_S_Matrix_Generator(int K, int S, char **A);

__global__ void I_H_Matrix_Generator(int K, int S, int H, char **A);

__global__ void G_LT_Matrix_Generator(int K, int S, int H, int L, int LP, char **A, int *ESIs, int M, uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1);

char** Matrix_A_Generator(Raptor10 &param, int* ESIs, int N);

void print_matrix(int row, int col, char **A);

void create_random_table_in_device(uint32_t *J, uint32_t *V0, uint32_t *V1);

void gaussianElimination(char **A, char **D, int numRows, int numACols, int numDCols, const int num_threads);

__global__ void init_D(int L, int K, int T, char **D, char **C_prime);

__global__ void LTEnc(int K, int S, int H, int T, int LP, int M, int* ESIs, char **C, char ** symbols_container, uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1);

void random_loss(int* ESIs, char** encoded_data, int N);

__global__ void check_result(char** data, char** decoded_data, int K, int T);

#endif
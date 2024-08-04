#include "cuda_raptor_10.cuh"
#include <iostream>
#include "raptor_consts.h"
#include <vector>
#include <assert.h>

void allocate_test_pointer(word *p, int BytesCount)
{
  cudaMalloc((word **)&p, BytesCount);
  cudaMemset(p, 0, BytesCount);
}

void create_random_table_in_device(uint32_t *d_J, uint32_t *d_V0, uint32_t *d_V1)
{
  // 定义数组大小
  const size_t J_size = sizeof(J);
  const size_t V0_size = sizeof(V0);
  const size_t V1_size = sizeof(V1);

  // 在设备上分配内存
  cudaMalloc((void **)&d_J, J_size);
  cudaMalloc((void **)&d_V0, V0_size);
  cudaMalloc((void **)&d_V1, V1_size);

  // 将数据从主机内存复制到设备内存
  cudaMemcpy(d_J, J, J_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V0, V0, V0_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V1, V1, V1_size, cudaMemcpyHostToDevice);
}

void cudaR10_compute_params(cudaRaptorParam *obj)
{
  if (!obj->Al && !obj->K && !obj->Kmax && !obj->Kmin && !obj->Gmax)
    return;

  uint32_t X = floor(sqrt(2 * obj->K));
  for (; X * X < 2 * obj->K + X; X++)
    ;

  // S number of LDPC symbols
  for (obj->S = ceil(0.01 * obj->K) + X; !is_prime(obj->S); obj->S++)
    ;

  // H number of Half symbols
  // not use for now
  // for (obj->H = 1; choose(obj->H, ceil(obj->H / 2)) < obj->K + obj->S; obj->H++) ;
  obj->H = 0;

  // L number of intermediate symbols
  obj->L = obj->K + obj->S + obj->H;
}

void printCudaRaptorParam(const cudaRaptorParam &param)
{
  std::cout << "F: " << param.F << std::endl;
  std::cout << "W: " << param.W << std::endl;
  std::cout << "P: " << param.P << std::endl;
  std::cout << "Al: " << param.Al << std::endl;
  std::cout << "Kmax: " << param.Kmax << std::endl;
  std::cout << "Kmin: " << param.Kmin << std::endl;
  std::cout << "Gmax: " << param.Gmax << std::endl;
  std::cout << "T: " << param.T << std::endl;
  std::cout << "Z: " << param.Z << std::endl;
  std::cout << "N: " << param.N << std::endl;
  std::cout << "K: " << param.K << std::endl;
  std::cout << "L: " << param.L << std::endl;
  std::cout << "S: " << param.S << std::endl;
  std::cout << "H: " << param.H << std::endl;
  std::cout << "G: " << param.G << std::endl;
  if (param.C)
    std::cout << "C: " << param.C << std::endl;
  if (param.Cp)
    std::cout << "Cp: " << param.Cp << std::endl;
}

void showFirstNonGPU(word *d_y, int N)
{
  word *hostValue;
  hostValue = (word *)malloc(N);
  cudaMemcpy(hostValue, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    std::cout << hostValue[i] << " | ";
  }
  std::cout << std::endl;
  free(hostValue);
}

namespace device
{
  __device__ uint32_t gray_bits_generate(uint32_t i)
  {
    return i ^ (int)(floor((float)(i / 2)));
  }

  __device__ uint32_t non_zero_bits_count(uint32_t v)
  {
    uint8_t tmp = 0;
    uint32_t bit_count = 0;
    ;
    for (int i = 0; i < 32; i++)
    {
      tmp = (v >> i) & 0x01;
      if (tmp == 0x01)
      {
        bit_count++;
      }
    }
    return bit_count;
  }

  __device__ bool choose_gray_bit(uint32_t num, uint32_t g)
  {
    return (bool)((g >> num) & 0x01);
  }

  __device__ int is_prime(uint32_t n)
  {
    if (n <= 1)
      return 0;

    for (uint32_t i = 2; i * i <= n; i++)
      if (!(n % i))
        return 0;

    return 1;
  }

  __device__ uint32_t r10_Rand(uint32_t X, uint32_t i, uint32_t m, uint32_t *device_V0, uint32_t *device_V1)
  {
    return (device_V0[(X + i) % 256] ^ device_V1[((uint32_t)(X / 256) + i) % 256]) % m;
  }

  __device__ uint32_t r10_Deg(uint32_t v)
  {
    if (v < 10241)
      return 1;
    if (v < 491582)
      return 2;
    if (v < 712794)
      return 3;
    if (v < 831695)
      return 4;
    if (v < 948446)
      return 10;
    if (v < 1032189)
      return 11;
    if (v < 1048576)
      return 40;
    return -1;
  }

  __device__ void r10_Trip(uint32_t K, uint32_t LP, int X, uint32_t triple[3], uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1)
  {
    uint32_t Q = 65521;
    uint32_t A = (53591 + device_J[K - 4] * 997) % Q;
    uint32_t B = 10267 * (device_J[K - 4] + 1) % Q;
    uint32_t Y = (B + X * A) % Q;
    // r10_Rand is passed 2^^20 as required by the RFC5053
    uint32_t v = r10_Rand(Y, 0, (2 << 15) * (2 << 3), device_V0, device_V1);
    uint32_t d = r10_Deg(v);
    uint32_t a = 1 + r10_Rand(Y, 1, LP - 1, device_V0, device_V1);
    uint32_t b = r10_Rand(Y, 2, LP, device_V0, device_V1);

    triple[0] = d;
    triple[1] = a;
    triple[2] = b;
  }

  // swap two rows
  __global__ void check_and_swap(char **A, char **B, int k, int* target_index)
  {
    if (*target_index == -1){
      // for (int i = 0; i < 23; i++){
      //   for (int j = 0; j < 23; j++){
      //     printf("%d ", A[i][j]);
      //   }
      //   printf("\n");
      // }
      assert(0 && "can't find a non-zero element in the kth column");
    }

    char *tempA = A[k];
    A[k] = A[*target_index];
    A[*target_index] = tempA;

    char *tempB = B[k];
    B[k] = B[*target_index];
    B[*target_index] = tempB;
  }
}

__global__ void printRandomTable(const uint32_t *d_J, const uint32_t *d_V0, const uint32_t *d_V1)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < 10)
  {
    printf("J[%d] = %u, V0[%d] = %u, V1[%d] = %u\n", idx, d_J[idx], idx, d_V0[idx], idx, d_V1[idx]);
  }
}

__global__ void test_Modify_A(int L, char **A)
{
  for (int i = 0; i < L; i++)
  {
    A[i][i] = 1;
  }
}

__global__ void LDPC_Matrix_Generator(int K, int S, char **A)
{
  uint32_t a = 0;
  uint32_t b = 0;

  for (int i = 0; i < K; i++)
  {
    a = 1 + ((uint32_t)floor((double)(i / S)) % (S - 1));
    b = i % S;
    A[b][i] = 1;
    b = (b + a) % S;
    A[b][i] = 1;
    b = (b + a) % S;
    A[b][i] = 1;
  }
}

__global__ void HALF_Matrix_Generator(int K, int S, int H, int HP, char **A)
{

  uint32_t g;
  bool m = 0;
  uint32_t bit_count;
  uint32_t i = 1;

  for (int h = S; h < S + H; h++)
  {
    for (int j = 0; j < (K + S); j++)
    {
      while (1)
      {

        g = device::gray_bits_generate(i);
        bit_count = device::non_zero_bits_count(g);
        i++;

        if (bit_count != HP)
        {
          continue;
        }

        m = device::choose_gray_bit(h - S, g);

        A[h][j] = m;

        break;
      }
    }
    i = 0;
  }
}

__global__ void I_S_Matrix_Generator(int K, int S, char **A)
{
  for (int i = 0; i < S; i++)
  {
    A[i][i + K] = 1;
  }
}

__global__ void I_H_Matrix_Generator(int K, int S, int H, char **A)
{
  for (int i = 0; i < H; i++)
  {
    A[i + S][i + K + S] = 1;
  }
}

__global__ void G_LT_Matrix_Generator(int K, int S, int H, int L, int LP, char **A, int *ESIs, int N, uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1)
{
  for (int i = 0; i < N; i++)
  {
    int ESI = ESIs[i];
    uint32_t triple[3] = {0};
    device::r10_Trip(K, LP, ESI, triple, device_J, device_V0, device_V1);
    uint32_t d = triple[0];
    uint32_t a = triple[1];
    uint32_t b = triple[2];

    while (b >= L)
    {
      b = (b + a) % LP;
    }

    A[i + S + H][b] = 1;

    int min = (d - 1 < L - 1) ? d - 1 : L - 1;
    for (int j = 1; j <= min; j++)
    {
      b = (b + a) % LP;
      while (b >= L)
      {
        b = (b + a) % LP;
      }

      A[i + S + H][b] = 1;
    }
  }
}

void print_matrix(int row, int col, char **A)
{
  // Copy A matrix from device to host
  std::vector<std::vector<char>> _A_host(row, std::vector<char>(col, 0));
  for (int i = 0; i < row; ++i)
  {
    char *d_row;
    cudaMemcpy(&d_row, &A[i], sizeof(char *), cudaMemcpyDeviceToHost);
    cudaMemcpy(_A_host[i].data(), d_row, col * sizeof(char), cudaMemcpyDeviceToHost);
  }

  // Print A matrix
  for (int i = 0; i < row; ++i)
  {
    for (int j = 0; j < col; ++j)
    {
      std::cout << static_cast<int>(_A_host[i][j]) << " ";
    }
    std::cout << std::endl;
  }
}

char **Matrix_A_Generator(Raptor10 &param, int* ESIs, int N)
{
  // Generate A matrix
  std::vector<std::vector<char>> _A(N + param.S + param.H, std::vector<char>(param.L, 0));

  // Allocate device pointer array
  char **A;
  cudaMalloc(&A, (N + param.S + param.H) * sizeof(char *));

  // Allocate device memory for each row and copy data
  for (int i = 0; i < (N + param.S + param.H); ++i)
  {
    char *d_row;
    cudaMalloc(&d_row, param.L * sizeof(char));
    cudaMemcpy(d_row, _A[i].data(), param.L * sizeof(char), cudaMemcpyHostToDevice);
    // Copy device row pointer to device pointer array
    cudaMemcpy(&A[i], &d_row, sizeof(char *), cudaMemcpyHostToDevice);
  }

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
  G_LT_Matrix_Generator<<<1, 1>>>(param.K, param.S, param.H, param.L, param.LP, A, ESIs, N, d_J, d_V0, d_V1);
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(1);
  }

  return A;
}

__global__ void print_index(int* target_index){
  printf("target index = %d\n", *target_index);
}

__global__ void xor_row(char** A, int row1, int row2){
  int threadIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

  A[row2][threadIdx_x] ^= A[row1][threadIdx_x];
}

// step is the number of threads
__global__ void gaussianFindMax(int* target_index, char **A, int k, int num_rows){
  int threadIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx_x >= k){
    if(A[threadIdx_x][k] == 1){
      *target_index = threadIdx_x;
      return;
    }
  }
}

__global__ void gaussianEliminateRows(char **A, char **D, int k, int num_rows, int num_ACols, int num_DCols){
  int threadIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx_x != k && A[threadIdx_x][k] == 1){
    xor_row<<<num_ACols, 1>>>(A, k, threadIdx_x);
    xor_row<<<num_DCols, 1>>>(D, k, threadIdx_x);
  }
}

void gaussianElimination(char **A, char **D, int numRows, int numACols, int numDCols, const int num_threads)
{
  int* target_index;
  cudaMalloc((void**)&target_index, sizeof(int));
  cudaMemset(target_index, -1, sizeof(int));

  int minDim = (numRows < numACols) ? numRows : numACols;
  for (int k = 0; k < minDim; ++k)
  {
    // find the first non-zero element in the kth column
    gaussianFindMax<<<numRows, 1>>>(target_index, A, k, numRows);
    cudaDeviceSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    // check if there is no 1 in the kth column, if not, swap the kth row with the i_max row
    device::check_and_swap<<<1, 1>>>(A, D, k, target_index);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    // subtract the kth row from the other rows to make the kth column all zeros
    gaussianEliminateRows<<<numRows, 1>>>(A, D, k, numRows, numACols, numDCols);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    cudaMemset(target_index, -1, sizeof(int));

    // printf("A = \n");
    // print_matrix(numRows, numACols, A);
    // getchar();
    //   // Print A matrix
    // for (int i = 0; i < numRows; ++i)
    // {
    //   for (int j = 0; j < numACols; ++j)
    //   {
    //     printf("%d ",A[i][j]);
    //   }
    //   printf("\n");
    // }
    // printf("D = \n");
    // for (int i = 0; i < numRows; ++i)
    // {
    //   for (int j = 0; j < numDCols; ++j)
    //   {
    //     printf("%d ",D[i][j]);
    //   }
    //   printf("\n");
    // }
  }
}

__global__ void init_D(int L, int N, int T, char **D, char **C_prime)
{
  int threadIdx_x = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < N; ++i)
  {
    D[L - N + i][threadIdx_x] = C_prime[i][threadIdx_x];
  }
}

__global__ void LTEnc(int K, int S, int H, int T, int LP, int M, int* ESIs, char **C, char ** symbols_container, uint32_t *d_J, uint32_t *d_V0, uint32_t *d_V1)
{
  int threadIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int L = K + S + H;

  for (int i = 0; i < M; i++)
  {
    int ESI = ESIs[i];
    uint32_t triple[3] = {0};
    device::r10_Trip(K, LP, ESI, triple, d_J, d_V0, d_V1);
    uint32_t d = triple[0];
    uint32_t a = triple[1];
    uint32_t b = triple[2];

    while (b >= L)
    {
      b = (b + a) % LP;
    }

    symbols_container[ESI][threadIdx_x] ^= C[b][threadIdx_x];

    int min = (d - 1 < L - 1) ? d - 1 : L - 1;
    for (int j = 1; j <= min; j++)
    {
      b = (b + a) % LP;
      while (b >= L)
      {
        b = (b + a) % LP;
      }
      symbols_container[ESI][threadIdx_x] ^= C[b][threadIdx_x];
    }
  }
}

void random_loss(int* ESIs, char** encoded_data, int N){
    int S = N - 3;

    // Allocate new memory for ESIs
    int* new_ESIs_d;
    cudaMalloc((void**)&new_ESIs_d, S * sizeof(int));

    // Copy data from old ESIs to new ESIs
    cudaMemcpy(new_ESIs_d, ESIs + 3, S * sizeof(int), cudaMemcpyDeviceToDevice);

    // Free old ESIs memory
    cudaFree(ESIs);

    // Update ESIs pointer to new memory
    ESIs = new_ESIs_d;

    // Allocate new memory for encoded_data pointers
    char** new_encoded_data_d;
    cudaMalloc((void**)&new_encoded_data_d, S * sizeof(char*));

    // Copy pointers from old encoded_data to new encoded_data
    cudaMemcpy(new_encoded_data_d, *encoded_data + 3, S * sizeof(char*), cudaMemcpyDeviceToDevice);

    // Free old encoded_data pointers memory
    cudaFree(*encoded_data);

    // Update encoded_data pointer to new memory
    encoded_data = new_encoded_data_d;
}

__global__ void check_result(char** data, char** decoded_data, int K, int T){
    for (int idx = 0; idx < K; idx++)
    {
        for (int j = 0; j < T; j++)
        {
            assert(data[idx][j] == decoded_data[idx][j] && "Data mismatch!");
        }
    }
}
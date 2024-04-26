#include "cuda_raptor_10.cuh"
#include <iostream>
#include <cuda_runtime.h>


int factorial(int n) {
  int result = 1, i;

  for (i = 2; i <= n; i++)
    result *= i;

  return result;
}

int is_prime(uint32_t n) {
  if (n <= 1)
    return 0;

  for (uint32_t i = 2; i * i <= n; i++)
    if (!(n % i))
      return 0;

  return 1;
}

__device__ int device_is_prime(uint32_t n) {
  if (n <= 1)
    return 0;

  for (uint32_t i = 2; i * i <= n; i++)
    if (!(n % i))
      return 0;

  return 1;
}

int choose(int i, int j) {
  return (factorial(i)) / (factorial(j) * factorial(i - j));
}

void cudaR10_compute_params(cudaRaptorParam *obj) {
  if (!obj->Al && !obj->K && !obj->Kmax && !obj->Kmin && !obj->Gmax)
    return;

  uint32_t X = floor(sqrt(2 * obj->K));
  for (; X * X < 2 * obj->K + X; X++)
    ;

  // S number of LDPC symbols
  for (obj->S = ceil(0.01 * obj->K) + X; !is_prime(obj->S); obj->S++)
    ;

  // H number of Half symbols
  for (obj->H = 1; choose(obj->H, ceil(obj->H / 2)) < obj->K + obj->S; obj->H++)
    ;

  // L number of intermediate symbols
  obj->L = obj->K + obj->S + obj->H;
}

void printCudaRaptorParam(const cudaRaptorParam& param) {
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

void showFirstNonGPU(word* d_y, int N){
    word *hostValue;
    hostValue = (word *)malloc(N);
    cudaMemcpy(hostValue, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
    {
        std::cout << hostValue[i] << std::endl;
    }
    free(hostValue);
}

__device__ uint32_t r10_Rand(uint32_t X, uint32_t i, uint32_t m, uint32_t *device_V0, uint32_t *device_V1) {
  return (device_V0[(X + i) % 256] ^ device_V1[((uint32_t)(X / 256) + i) % 256]) % m;
}

__device__ uint32_t r10_Deg(uint32_t v) {
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

__device__ void r10_Trip(uint32_t K, uint32_t L, uint32_t X, uint32_t triple[3], uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1) {
  uint32_t L_ = L;
  while (!device_is_prime(L_))
    L_++;

  uint32_t Q = 65521;
  uint32_t A = (53591 + device_J[K - 4] * 997) % Q;
  uint32_t B = 10267 * (device_J[K - 4] + 1) % Q;
  uint32_t Y = (B + X * A) % Q;
  // r10_Rand is passed 2^^20 as required by the RFC5053
  uint32_t v = r10_Rand(Y, 0, (2 << 15) * (2 << 3), device_V0, device_V1);
  uint32_t d = r10_Deg(v);
  uint32_t a = 1 + r10_Rand(Y, 1, L_ - 1, device_V0, device_V1);
  uint32_t b = r10_Rand(Y, 2, L_, device_V0, device_V1);

  triple[0] = d;
  triple[1] = a;
  triple[2] = b;
}

__global__ void cudaLTEnc(const int K, word *C, word *EncC, const int L, const int N, uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;

    if (id >= N) return;

    uint32_t L_ = L;
    while (!device_is_prime(L_))
        L_++;

    uint32_t triple[3] = {0};
    r10_Trip(K, L, id, triple, device_J, device_V0, device_V1);
    uint32_t d = triple[0];
    uint32_t a = triple[1];
    uint32_t b = triple[2];
    uint32_t j_max = min((d - 1), (L - 1));

    while (b >= L)
      b = (b + a) % L_;

    word result = C[b];
    for (uint j = 1; j <= j_max; j++) {
      b = (b + a) % L_;

      while (b >= L)
        b = (b + a) % L_;

      result = result ^ C[b];
      EncC[id] = result;
    }
}
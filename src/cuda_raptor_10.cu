#include "cuda_raptor_10.cuh"
#include <iostream>

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

void showFirstNonGPU(float* d_y, int N){
    float *hostValue;
    hostValue = (float *)malloc(N);
    cudaMemcpy(hostValue, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
    {
        std::cout << hostValue[i] << std::endl;
    }
    free(hostValue);
}

__global__ void cudaLTEnc(float *A, float *B, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;

    printf("ID: %d - ", id);
    printf("%f\n", B[id]);
    B[id] = id * 1.0;
    // C[id] = A[id] + B[id];
}
#ifndef CUDA_RAPTOR_10_CUH
#define CUDA_RAPTOR_10_CUH

#include <stdint.h>
#include <cuda_raptor_10.h>

__global__ void cudaLTEncImpl(const int K, word *C, word *EncC, const int L, const int N, uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1);

#endif
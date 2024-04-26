#ifndef CUDA_RAPTOR_10_H
#define CUDA_RAPTOR_10_H

#include <stdint.h>

/** Typedef word: here as a uint32_t type */
typedef uint32_t word;

typedef struct {
  uint32_t F;
  uint32_t W;
  uint32_t P;
  uint32_t Al;
  uint32_t Kmax;
  uint32_t Kmin;
  uint32_t Gmax;
  uint32_t T;
  uint32_t Z;
  uint32_t N;
  uint32_t K;
  uint32_t L;
  uint32_t S;
  uint32_t H;
  uint32_t G;
  uint8_t *C;
  uint8_t *Cp;
} cudaRaptorParam;

/**
 * Function responsible for computing all the needed parameters.
 * @param params_obj Raptor10 object to configure
 */
void cudaR10_compute_params(cudaRaptorParam *obj);

void printCudaRaptorParam(const cudaRaptorParam& obj);

void showFirstNonGPU(word* d_y, int N);

void allocate_test_pointer(word* p, int BytesCount);

void create_random_table_in_device(uint32_t* J, uint32_t* V0, uint32_t* V1);

__global__ void cudaLTEnc(const int K, word *C, word *EncC, const int L, const int N, uint32_t *device_J, uint32_t *device_V0, uint32_t *device_V1);

#endif
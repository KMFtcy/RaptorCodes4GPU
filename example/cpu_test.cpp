#include <iostream>
#include <stdint.h>
#include "raptor10.hpp"
#include "gf2matrix.hpp"

using namespace std;

int main(){
  // Create a Raptor10 object and fill it w/ all known needed params
  Raptor10 coder;
  unsigned int K = 1024;

  // G = min{ceil(P*Kmin/F), P/Al, Gmax}
  // T = floor(P/(Al*G))*Al
  // Kt = ceil(F/T)
  // Z = ceil(Kt/Kmax)
  // N = min{ceil(ceil(Kt/Z)*T/W), T/Al}
  coder.K = K;
  coder.Kmin = 1024; //a minimum target on the number of symbols per source block
  coder.Kmax = 8192; //the maximum number of source symbols per source block.
  coder.Gmax = 10; //a maximum target number of symbols per packet
  coder.Al = 4; //the symbol alignment parameter, in bytes, 一个symbol的长度
  coder.N = coder.K + 4;
  coder.T = 4; // symbol size, in bytes
  r10_compute_params(&coder);
  printf("K=%u, S=%u, H=%u, L=%u\n", coder.K, coder.S, coder.H, coder.L);

  // Allocate and calculate the constraints matrix
  gf2matrix A;
  allocate_gf2matrix(&A, coder.L, coder.L);
  int build_result = r10_build_constraints_mat(&coder, &A);
  if (build_result != 0){
    return build_result;
  }
  printf("Built constraints matrix.\n");

  // Allocate and calculate the LT code matrix
  gf2matrix G_LT;
  allocate_gf2matrix(&G_LT, coder.L, coder.N); // Calculate the LT matrix and encoded symbols
  uint32_t ESIs[coder.N]; // Create vector of ESIs
  for (uint32_t i = 0; i < coder.N; i++)
    ESIs[i] = i;
  r10_build_LT_mat(coder.N, &coder, &G_LT, ESIs);
  printf("Built LT matrix.\n");


  // LT encode
  uint8_t enc_s[coder.L * coder.T];
  uint8_t src_s[coder.K * coder.T];
  clock_t start,end;
  printf("Start coding.\n");
  start = clock();
  r10_encode(src_s, enc_s, &coder, &A, &G_LT, ESIs);
  end = clock();
  printf("Encoded.\n");

  // Now, enc_s should contain the encoded symbols
  // Still, doesn't allow to decide the size of the symbols
  double running_time = (double)(end-start)/CLOCKS_PER_SEC;
  int data_size = coder.K * coder.T;
  printf("Coding time:%fs\n", running_time);
  printf("Coding rate:%dkB/s\n", (int)(data_size/(1000*running_time)));
  // printf("Constraints matrix:\n");
  // print_matrix(&A);
}
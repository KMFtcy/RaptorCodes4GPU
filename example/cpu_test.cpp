#include <iostream>
#include <stdint.h>
#include <vector>
#include <math.h>
#include "raptor10.hpp"

using namespace std;


vector<char>* multiply_symbols(vector<char>& A, vector<char>& B) {
    if (A.size() != B.size()) {
        return NULL;
    }

    int size = A.size();
    vector<char>* C = new vector<char>(size, 0);
    for (int i = 0; i < size; i++) {
        (*C)[i] = A[i] ^ B[i];
    }
    return C;
}

int main()
{
    cout << "CPU test begin" << endl;
    // Initialize paramaters
    Raptor10 params;
    params.Kmin = 1024; // a minimum target on the number of symbols per source block
    params.Kmax = 8192; // the maximum number of source symbols per source block.
    params.Gmax = 10;   // a maximum target number of symbols per packet
    params.T = 1400;    // symbol size, suppose to be a ip packet size
    params.K = 1024;
    params.Al = 4; // the symbol alignment parameter, in bytes, 一个symbol的长度
    int L_ = params.L;
    r10_compute_params(&params);
    params.N = params.K + params.S + 10;
    cout << "K = " << params.K;
    cout << ", S = " << params.S;
    cout << ", H = " << params.H;
    cout << ", L = " << params.L ;
    cout << ", N = " << params.N << endl;
    // LT coding params
    while (!is_prime(L_))
        L_++;
    vector<int> ESIs(params.N); // Create vector of ESIs
    for (int i = 0; i < params.N; i++){
        ESIs[i] = i;
    }

    int origin_bytes_count = params.T * params.K;
    int coded_bytes_count = params.T * params.L;

    // prepare container for data and encoded data
    vector<vector<char>> data(params.K, vector<char>(params.T, 1));
    vector<vector<char>> encoded_data(params.N, vector<char>(params.T, 1));
    vector<vector<char>> LT_data(params.N, vector<char>(params.T, 1));

    for (int i = 0; i < params.K; i++)
    {
        for (int j = 0; j < params.T; j++)
        {
            data[i][j] = rand();
            encoded_data[i][j] = data[i][j];
        }
    }

    cout << "Data: " << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << (int)data[i][0] << " | ";
    }
    cout << endl;

    // start LDPC
    clock_t start, end;
    int a = 0, b = 0;
    start = clock();
    // LDPC coding
    for (int i = 0; i < params.K; i++)
    {
        a = 1 + ((int)floor(i / params.S) % (params.S - 1));
        b = i % params.S;
        encoded_data[params.K + b] = *multiply_symbols(encoded_data[i], encoded_data[params.K + b]);
        b = (b + a) % params.S;
        encoded_data[params.K + b] = *multiply_symbols(encoded_data[i], encoded_data[params.K + b]);
        b = (b + a) % params.S;
        encoded_data[params.K + b] = *multiply_symbols(encoded_data[i], encoded_data[params.K + b]);
    }
    // LT coding
    for (uint32_t i = params.L; i < params.N; i++) {
        uint32_t triple[3] = {0};
        uint32_t X = ESIs[i];
        r10_Trip(params.K, X, triple, &params);
        uint32_t d = triple[0];
        uint32_t a = triple[1];
        uint32_t b = triple[2];
        uint32_t j_max = fmin((d - 1), (params.L - 1));

        while (b >= params.L){
          b = (b + a) % L_;
        }

        encoded_data[i] = encoded_data[b];
        // set_entry(G_LT, i, b, 1);

        for (int j = 1; j <= j_max; j++) {
          b = (b + a) % L_;

          while (b >= params.L)
              b = (b + a) % L_;

          encoded_data[i] = *multiply_symbols(encoded_data[i], encoded_data[b]);
        }
    }
    end = clock();
    double running_time = (double)(end - start) / CLOCKS_PER_SEC;
    int data_size = origin_bytes_count;
    printf("Coding time:%fs\n", running_time);
    printf("Coding rate:%dMB/s\n", (int)(data_size / (1000000 * running_time)));

}
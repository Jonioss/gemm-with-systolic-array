#include "constants.h"

void load_A(hls::burst_maxi<hls::vector<float, VEC_SIZE> A_DRAM, float A_BUF[I][K]) {
    #pragma HLS INLINE off
    for(int i = 0; i < I; i++) {
        for(int k = 0; k < K/VEC_SIZE; k++) {
            #pragma HLS PIPELINE II=1
            hls::vector<float, VEC_SIZE> temp = A_DRAM.read();
            for(int v = 0; v < VEC_SIZE; v++) {
                #pragma HLS UNROLL
                A_BUF[i][k*VEC_SIZE + v] = temp[v];
            }
        }
    }
}

void load_B(hls::burst_maxi<hls::vector<float, VEC_SIZE> B_DRAM, float B_BUF[K][J]) {
    #pragma HLS INLINE off
    for(int k = 0; k < K/VEC_SIZE; k++) {
        for(int j = 0; j < J; j++) {
            #pragma HLS PIPELINE II=1
            hls::vector<float, VEC_SIZE> temp = B_DRAM.read();
            for(int v = 0; v < VEC_SIZE; v++) {
                #pragma HLS UNROLL
                B_BUF[k*VEC_SIZE + v][j] = temp[v];
            }
        }
    }
}

void loadInputsFromDRAM(hls::burst_maxi<hls::vector<float, VEC_SIZE> A_DRAM, hls::burst_maxi<hls::vector<float, VEC_SIZE> B_DRAM, float A_BUF[I][K], float B_BUF[K][J]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    load_A(A_DRAM, A_BUF);
    load_B(B_DRAM, B_BUF);
}

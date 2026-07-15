#include "constants.h"

void load_A(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, float A_BUF[I][K]) {
    #pragma HLS INLINE off
    A_DRAM.read_request(0, I*K/VEC_SIZE);
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

void load_B(hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, float B_BUF[K][J]) {
    #pragma HLS INLINE off
    B_DRAM.read_request(0, K*J/VEC_SIZE);
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

void loadInputsFromDRAM(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, float A_BUF[I][K], float B_BUF[K][J]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    load_A(A_DRAM, A_BUF);
    load_B(B_DRAM, B_BUF);
}

void storeOutputToDRAM(float C_BUF[I][J], hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM) {
    #pragma HLS INLINE off
    C_DRAM.write_request(0, I*J/VEC_SIZE);
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J/VEC_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN
            hls::vector<float, VEC_SIZE> temp;
            for(int v = 0; v < VEC_SIZE; v++) {
                #pragma HLS UNROLL
                temp[v] = C_BUF[i][j*VEC_SIZE + v];
            }
            C_DRAM.write(temp);
        }
    }
    C_DRAM.write_response();
}


#include "constants.h"

void A_Buf_to_stream(float A_BUF[I][K], hls::stream<float> A_stream[S_A_I][S_A_J+1]) {
    #pragma HLS INLINE off
    for(int i = 0; i < S_A_I; i++) {
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            A_stream[i][0].write(A_BUF[i][k]);
        }
    }
}

void B_Buf_to_stream(float B_BUF[K][J], hls::stream<float> B_stream[S_A_J+1][S_A_I]) {
    #pragma HLS INLINE off
    for(int j = 0; j < S_A_J; j++) {
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            B_stream[0][j].write(B_BUF[k][j]);
        }
    }
}

void load_tile_A(float A_BUF[I][K], float A_TILE[S_A_I][K]) {
    #pragma HLS INLINE off


}

void tm_A(float A_BUF[I][K], hls::stream<float> &A_stream) {
    #pragma HLS INLINE off

    float A_TILE[2][S_A_I][K];
    #pragma HLS STREAM type=pipo variable=A_TILE depth=2

    // load A to A_TILE
    // send A_TILE to A_stream to be then fed into the systolic array
}
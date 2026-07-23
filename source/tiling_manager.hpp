#include "constants.h"

void stream_to_C_Buf(hls::stream<float> C_stream[S_A_I][S_A_J], float C_BUF[I][J], int tileC) {
    #pragma HLS INLINE off
    for (int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < S_A_J; j++) {
            #pragma HLS UNROLL
            C_BUF[tileC*S_A_I + i][j] = C_stream[i][j].read();
        }
    }
}

void B_Buf_to_stream(float B_BUF[K][J/NUM_TILES_J], hls::stream<float> B_stream[S_A_J+1][S_A_I]) {
    #pragma HLS INLINE off
    for(int j = 0; j < S_A_J; j++) {
        #pragma HLS UNROLL
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            B_stream[0][j].write(B_BUF[k][j]);
        }
    }
}

void load_tile_A(float A_BUF[I][K], float A_TILE[S_A_I][K], int tileA) {
    #pragma HLS INLINE off

    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            A_TILE[i][k] = A_BUF[tileA*S_A_I + i][k];
        }
    }
}

void tile_A_to_stream(float A_TILE[S_A_I][K], hls::stream<float> A_stream[S_A_I][S_A_J+1]) {
    #pragma HLS INLINE off
    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            A_stream[i][0].write(A_TILE[i][k]);
        }
    }
}

void tm_A(float A_BUF[I][K], hls::stream<float> A_stream[S_A_I][S_A_J+1], int tileA) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
 
    static float A_TILE[S_A_I][K];
    #pragma HLS STREAM variable=A_TILE type=pipo depth=2
    #pragma HLS ARRAY_PARTITION variable=A_TILE type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A_TILE type=complete dim=2

    load_tile_A(A_BUF, A_TILE, tileA);
    tile_A_to_stream(A_TILE, A_stream);
}
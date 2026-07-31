#include "constants.h"

void stream_to_C_Vec(hls::stream<float> C_stream[S_A_I][S_A_J], hls::stream<hls::vector<float, S_A_J>> &C_out) {
    #pragma HLS INLINE off
    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS PIPELINE II=1
        hls::vector<float, S_A_J> temp;
        for(int j = 0; j < S_A_J; j++) {
            #pragma HLS UNROLL
            temp[j] = C_stream[i][j].read();
        }
        C_out.write(temp);
    }
}

void B_Vec_to_Buf(hls::stream<hls::vector<float, K>> &B_in, float B_BUF[K][J]) {
    #pragma HLS INLINE off
    for(int j = 0; j < J; j++) {
        #pragma HLS PIPELINE II=1
        hls::vector<float, K> temp = B_in.read();
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            B_BUF[k][j] = temp[k];
        }
    }
}

void B_Buf_to_stream(float B_BUF[K][J], hls::stream<float> B_stream[S_A_J+1][S_A_I]) {
    #pragma HLS INLINE off
    for(int k = 0; k < K; k++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < J; j++) {
            #pragma HLS UNROLL
            B_stream[0][j].write(B_BUF[k][j]);
        }
    }
}

void load_tile_A(hls::stream<hls::vector<float, K>> &A_in, float A_TILE[S_A_I][K]) {
    #pragma HLS INLINE off
    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS PIPELINE II=1
        hls::vector<float, K> temp = A_in.read();
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            A_TILE[i][k] = temp[k];
        }
    }
}

void tile_A_to_stream(float A_TILE[S_A_I][K], hls::stream<float> A_stream[S_A_I][S_A_J+1]) {
    #pragma HLS INLINE off
    for(int k = 0; k < K; k++) {
        #pragma HLS PIPELINE II=1
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS UNROLL
            A_stream[i][0].write(A_TILE[i][k]);
        }
    }
}

void tm_A(hls::stream<hls::vector<float, K>> &A_in, hls::stream<float> A_stream[S_A_I][S_A_J+1]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
 
    float A_TILE[S_A_I][K];
    #pragma HLS STREAM variable=A_TILE type=pipo depth=2
    #pragma HLS ARRAY_PARTITION variable=A_TILE type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A_TILE type=complete dim=2

    load_tile_A(A_in, A_TILE);
    tile_A_to_stream(A_TILE, A_stream);
}

// void tm_B(hls::stream<hls::vector<float, K>> &B_in, hls::stream<float> B_stream[S_A_J+1][S_A_I]) {
//     #pragma HLS INLINE off
//     #pragma HLS DATAFLOW

//     float B_BUF[K][J];
//     #pragma HLS STREAM variable=B_BUF type=shared
//     #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=2

//     B_Vec_to_Buf(B_in, B_BUF);
//     B_Buf_to_stream(B_BUF, B_stream);
// }
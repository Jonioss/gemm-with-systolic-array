#include "constants.h"
#include "tiling_manager.hpp"

void sink_streams(hls::stream<float> A_stream[S_A_I][S_A_J+1], hls::stream<float> B_stream[S_A_J+1][S_A_I]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    sink_a:
    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            (void) A_stream[i][S_A_J].read();
        }
    }
 
    sink_b:
    for(int j = 0; j < S_A_J; j++) {
        #pragma HLS UNROLL
        for(int k = 0; k < K; k++) {
            #pragma HLS UNROLL
            (void) B_stream[S_A_I][j].read();
        }
    }
}

// void matmul(PE sar[S_A_I][S_A_J], 
// hls::stream<float> A_stream[S_A_I][S_A_J+1], hls::stream<float> B_stream[S_A_J+1][S_A_I], hls::stream<float> C_stream[S_A_I][S_A_J]) {
//     #pragma HLS INLINE off
//     for(int i = 0; i < S_A_I; i++) {
//         #pragma HLS UNROLL  
//         for(int j = 0; j < S_A_J; j++) {
//             #pragma HLS UNROLL
//             sar[i][j].run(A_stream[i][j], B_stream[i][j], A_stream[i][j+1], B_stream[i+1][j], C_stream[i][j]);
//         }
//     }
// }

void runSystolicArray(float A_BUF[I][K], float B_BUF[K][J], float C_BUF[I][J], PE sar[S_A_I][S_A_J]) {
    #pragma HLS INLINE off

    hls::stream<float> A_stream[S_A_I][S_A_J+1];
    #pragma HLS STREAM variable=A_stream type=fifo depth=4
    hls::stream<float> B_stream[S_A_J+1][S_A_I];
    #pragma HLS STREAM variable=B_stream type=fifo depth=4
    hls::stream<float> C_stream[S_A_I][S_A_J];

    tile_loop:
    for (int t = 0; t < NUM_TILES_I; t++) {
        #pragma HLS DATAFLOW

        tm_A(A_BUF, A_stream, t);
        B_Buf_to_stream(B_BUF, B_stream);
 
        calculate_matmul:
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS UNROLL  
            for(int j = 0; j < S_A_J; j++) {
                #pragma HLS UNROLL
                sar[i][j].run(A_stream[i][j], B_stream[i][j], A_stream[i][j+1], B_stream[i+1][j], C_stream[i][j]);
            }
        }
 
        sink_streams(A_stream, B_stream);
 
        stream_to_C_Buf(C_stream, C_BUF, t);
    }
}
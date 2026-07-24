#include "constants.h"
#include "tiling_manager.hpp"

void sink_streams(hls::stream<float> A_stream[S_A_I][S_A_J+1], hls::stream<float> B_stream[S_A_J+1][S_A_I]) {
    #pragma HLS INLINE off

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

void runSystolicArray(float A_BUF[I][K], float B_BUF[NUM_TILES_J][K][S_A_J], float C_BUF[NUM_TILES_I][NUM_TILES_J][S_A_I][S_A_J]) {
    #pragma HLS INLINE off
 
    tile_loop_A:
    for(int tileA = 0; tileA < NUM_TILES_I; tileA++) {
        #pragma HLS DATAFLOW
 
        hls::stream<float> A_stream[NUM_TILES_J][S_A_I][S_A_J+1];
        #pragma HLS STREAM variable=A_stream type=fifo depth=K
        hls::stream<float> B_stream[NUM_TILES_J][S_A_J+1][S_A_I];
        #pragma HLS STREAM variable=B_stream type=fifo depth=K
        hls::stream<float> C_stream[NUM_TILES_J][S_A_I][S_A_J];
        #pragma HLS STREAM variable=C_stream type=fifo depth=K
 
        tm_A(A_BUF, A_stream, tileA);
 
        tile_loop_B:
        for (int tileB = 0; tileB < NUM_TILES_J; tileB++) {
            #pragma HLS UNROLL
 
            B_Buf_to_stream(B_BUF[tileB], B_stream[tileB]);
 
            calculate_matmul:
            for(int i = 0; i < S_A_I; i++) {
                #pragma HLS UNROLL
                for(int j = 0; j < S_A_J; j++) {
                    #pragma HLS UNROLL
                    PE(A_stream[tileB][i][j], B_stream[tileB][i][j], A_stream[tileB][i][j+1], B_stream[tileB][i+1][j], C_stream[tileB][i][j]);
                }
            }
 
            sink_streams(A_stream[tileB], B_stream[tileB]);
 
            stream_to_C_Buf(C_stream[tileB], C_BUF, tileA, tileB);
        }
    }
}

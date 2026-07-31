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

void computeSystolicArray(hls::stream<float> A_stream[S_A_I][S_A_J+1], hls::stream<float> B_stream[S_A_J+1][S_A_I],
hls::stream<float> C_stream[S_A_I][S_A_J]) {
    #pragma HLS INLINE

    calculate_matmul:
    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for(int j = 0; j < S_A_J; j++) {
            #pragma HLS UNROLL
            PE(A_stream[i][j], B_stream[i][j], A_stream[i][j+1], B_stream[i+1][j], C_stream[i][j]);
        }
    }
}

void runSystolicArray(hls::stream<hls::vector<float, K>> A_in[NUM_TILES_I], 
    hls::stream<hls::vector<float, K>> B_in[NUM_TILES_J], 
    hls::stream<hls::vector<float, S_A_J>> C_out[NUM_TILES_I][NUM_TILES_J]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    float B_TILE[K][S_A_J];
    #pragma HLS STREAM variable=B_TILE type=pipo
    #pragma HLS ARRAY_PARTITION variable=B_TILE type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_TILE type=complete dim=2

    tile_loop_B:
    for(int tileB = 0; tileB < NUM_TILES_J; tileB++) {

        B_Vec_to_Buf(B_in[tileB], B_TILE);
        
        tile_loop_A:
        for(int tileA = 0; tileA < NUM_TILES_I; tileA++) {
            #pragma HLS DATAFLOW
            #pragma HLS STABLE variable=B_TILE
    
            hls::stream<float> A_stream[S_A_I][S_A_J+1];
            #pragma HLS STREAM variable=A_stream type=fifo depth=K
            hls::stream<float> B_stream[S_A_J+1][S_A_I];
            #pragma HLS STREAM variable=B_stream type=fifo depth=K
            hls::stream<float> C_stream[S_A_I][S_A_J];
            #pragma HLS STREAM variable=C_stream type=fifo depth=K

            tm_A(A_in[tileA], A_stream);
            B_Buf_to_stream(B_TILE, B_stream);
            computeSystolicArray(A_stream, B_stream, C_stream);
            sink_streams(A_stream, B_stream);
            stream_to_C_Vec(C_stream, C_out[tileA][tileB]);
        }
    }
}

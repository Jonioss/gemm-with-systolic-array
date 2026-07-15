#include "constants.h"
#include "tiling_manager.hpp"

void clear_sa(PE systolic_array[S_A_I][S_A_J]) {
    #pragma HLS INLINE off
    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for(int j = 0; j < S_A_J; j++) {
            systolic_array[i][j].clear_acc();
        }
    }
}

void runSystolicArray(float A_BUF[I][K], float B_BUF[K][J], float C_BUF[I][J], PE sar[S_A_I][S_A_J]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
 
    // Local now, not arguments -- this is what makes DATAFLOW checking
    // pass. A_BUF/B_BUF/C_BUF are the true inputs/outputs of this
    // function (read once at the start by the feed tasks, written once
    // at the end by the drain task); A_stream/B_stream/C_stream are
    // purely internal wiring between the PE tasks.
    hls::stream<float> A_stream[S_A_I][S_A_J+1];
    hls::stream<float> B_stream[S_A_J+1][S_A_I];
    hls::stream<float> C_stream[S_A_I][S_A_J];
 
    A_Buf_to_stream(A_BUF, A_stream);
    B_Buf_to_stream(B_BUF, B_stream);
 
    calculate_matmul:
    for(int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for(int j = 0; j < S_A_J; j++) {
            #pragma HLS UNROLL
            sar[i][j].run(A_stream[i][j], B_stream[i][j], A_stream[i][j+1], B_stream[i+1][j], C_stream[i][j]);
        }
    }
 
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
 
    stream_to_C_Buf(C_stream, C_BUF);
}
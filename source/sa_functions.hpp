#include "constants.h"

void clear_sa(PE systolic_array[S_A_I][S_A_J]) {
    #pragma HLS INLINE off
    for(int i = 0; i < S_A_I; i++) {
        for(int j = 0; j < S_A_J; j++) {
            systolic_array[i][j].clear_acc();
        }
    }
}

void runSystolicArray(hls::stream<float> A_stream[S_A_I][S_A_J+1], hls::stream<float> B_stream[S_A_J+1][S_A_I],
    hls::stream<float> C_stream[S_A_I][S_A_J], PE sar[S_A_I][S_A_J]) {
        #pragma HLS INLINE off
        #pragma HLS DATAFLOW

        calculate_matmul:
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS UNROLL
            for(int j = 0; j < S_A_J; j++) {
                sar[i][j].run(A_stream[i][j], B_stream[i][j], A_stream[i][j+1], B_stream[i+1][j], C_stream[i][j]);
            }
        }

        sink_a:
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS UNROLL
            for(int k = 0; k < K; k++) {
                (void) A_stream[i][S_A_J].read();
            }
        }

        sink_b:
        for(int j = 0; j < S_A_J; j++) {
            #pragma HLS UNROLL
            for(int k = 0; k < K; k++) {
                (void) B_stream[S_A_I][j].read();
            }
        }
    }


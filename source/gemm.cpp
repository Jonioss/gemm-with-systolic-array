#include "constants.h"
#include "dram.hpp"
#include "pe.h"
#include "sa_functions.hpp"
#include "tiling_manager.hpp"

void gemm(const hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, 
    hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM) {

	#pragma HLS INTERFACE m_axi offset=slave port=A_DRAM bundle=gmem0 depth=I*K/VEC_SIZE max_read_burst_length=VEC_SIZE
	#pragma HLS INTERFACE m_axi offset=slave port=B_DRAM bundle=gmem1 depth=K*J/VEC_SIZE max_read_burst_length=VEC_SIZE
	#pragma HLS INTERFACE m_axi offset=slave port=C_DRAM bundle=gmem2 depth=I*J/VEC_SIZE max_write_burst_length=VEC_SIZE

    float A_BUF[I][K];
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=block factor=NUM_TILES_I dim=1
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=complete dim=2

    float B_BUF[K][J];
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=block factor=NUM_TILES_J dim=2

    float C_BUF[I][J];
    #pragma HLS ARRAY_PARTITION variable=C_BUF type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C_BUF type=complete dim=2
    
    hls::stream<float> A_stream[S_A_I][S_A_J+1];
    hls::stream<float> B_stream[S_A_J+1][S_A_I];
    hls::stream<float> C_stream[S_A_I][S_A_J];

    PE systolic_array[S_A_I][S_A_J];
    #pragma HLS ARRAY_PARTITION variable=systolic_array type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=systolic_array type=complete dim=2

    loadInputsFromDRAM(A_DRAM, B_DRAM, A_BUF, B_BUF);

    A_Buf_to_stream(A_BUF, A_stream);
    B_Buf_to_stream(B_BUF, B_stream);

    runSystolicArray(A_stream, B_stream, C_stream, systolic_array);

    drain_c:
    for (int i = 0; i < S_A_I; i++) {
        #pragma HLS UNROLL
        for (int j = 0; j < S_A_J; j++) {
            C_BUF[i][j] = C_stream[i][j].read();
        }
    }

    storeOutputToDRAM(C_BUF, C_DRAM);

}

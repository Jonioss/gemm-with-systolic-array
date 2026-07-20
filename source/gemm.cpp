#include "constants.h"
#include "dram.hpp"
#include "pe.h"
#include "sa_functions.hpp"

void gemm(const hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, 
    hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM) {
 
	#pragma HLS INTERFACE m_axi offset=slave port=A_DRAM bundle=gmem0 depth=I*K/VEC_SIZE max_read_burst_length=std::min(VEC_SIZE, 16)
	#pragma HLS INTERFACE m_axi offset=slave port=B_DRAM bundle=gmem1 depth=K*J/VEC_SIZE max_read_burst_length=std::min(VEC_SIZE, 16)
	#pragma HLS INTERFACE m_axi offset=slave port=C_DRAM bundle=gmem2 depth=I*J/VEC_SIZE max_write_burst_length=std::min(VEC_SIZE, 16)
 
    float A_BUF[I][K];
    #pragma HLS BIND_STORAGE variable=A_BUF type=RAM_1WNR impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=block factor=NUM_TILES_I dim=1
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=complete dim=2
 
    float B_BUF[K][J];
    // #pragma HLS BIND_STORAGE variable=B_BUF type=RAM_1WNR impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=2
 
    float C_BUF[I][J];
    #pragma HLS ARRAY_PARTITION variable=C_BUF type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C_BUF type=complete dim=2
 
    PE systolic_array[S_A_I][S_A_J];
    #pragma HLS ARRAY_PARTITION variable=systolic_array type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=systolic_array type=complete dim=2

    loadInputsFromDRAM(A_DRAM, B_DRAM, A_BUF, B_BUF);
    runSystolicArray(A_BUF, B_BUF, C_BUF, systolic_array);
    storeOutputToDRAM(C_BUF, C_DRAM);
}
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
    #pragma HLS STREAM variable=A_BUF type=fifo
    #pragma HLS BIND_STORAGE variable=A_BUF type=RAM_S2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=block factor=NUM_TILES_I dim=1
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=complete dim=2
 
    float B_BUF[NUM_TILES_J][K][J/NUM_TILES_J];
    #pragma HLS STREAM variable=B_BUF type=fifo
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=3

    float C_BUF[I][J];
    #pragma HLS STREAM variable=C_BUF type=fifo
    #pragma HLS ARRAY_PARTITION variable=C_BUF type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C_BUF type=complete dim=2

    // #pragma HLS DATAFLOW

    loadInputsFromDRAM(A_DRAM, B_DRAM, A_BUF, B_BUF);
    runSystolicArray(A_BUF, B_BUF, C_BUF);
    storeOutputToDRAM(C_BUF, C_DRAM);
}
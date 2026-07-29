#include "constants.h"
#include "dram.hpp"
#include "pe.h"
#include "sa_functions.hpp"

void gemm(const hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, 
    hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM) {
 
	#pragma HLS INTERFACE m_axi offset=slave port=A_DRAM bundle=gmem0 depth=I*K/VEC_SIZE max_read_burst_length=std::min(VEC_SIZE, 16) latency=64
	#pragma HLS INTERFACE m_axi offset=slave port=B_DRAM bundle=gmem1 depth=K*J/VEC_SIZE max_read_burst_length=std::min(VEC_SIZE, 16) latency=64
	#pragma HLS INTERFACE m_axi offset=slave port=C_DRAM bundle=gmem2 depth=I*J/VEC_SIZE max_write_burst_length=std::min(VEC_SIZE, 16) latency=64

    hls::stream<hls::vector<float, S_A_J>> C_out[NUM_TILES_I];
    #pragma HLS STREAM variable=C_out type=fifo depth=S_A_I

    hls::stream<hls::vector<float, K>> A_in[NUM_TILES_I];
    #pragma HLS STREAM variable=A_in type=fifo depth=S_A_I

    hls::stream<hls::vector<float, K>> B_in;
    #pragma HLS STREAM variable=B_in type=fifo depth=J

    #pragma HLS DATAFLOW

    loadInputsFromDRAM(A_DRAM, B_DRAM, A_in, B_in);
    runSystolicArray(A_in, B_in, C_out);
    storeOutputToDRAM(C_out, C_DRAM);
}
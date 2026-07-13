#include "constants.h"

void gemm(const hls::burst_maxi<hls::vector<float, VEC_SIZE> A_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE> B_DRAM, 
    hls::burst_maxi<hls::vector<float, VEC_SIZE> C_DRAM>) {

    // HLS INTERFACE pragmas for AXI4 master interfaces
	#pragma HLS INTERFACE m_axi offset=slave port=A_DRAM bundle=gmem0 depth=I*K/VEC_SIZE max_read_burst_length=VEC_SIZE
	#pragma HLS INTERFACE m_axi offset=slave port=B_DRAM bundle=gmem1 depth=K*J/VEC_SIZE max_read_burst_length=VEC_SIZE
	#pragma HLS INTERFACE m_axi offset=slave port=C_DRAM bundle=gmem2 depth=I*J/VEC_SIZE max_write_burst_length=VEC_SIZE

    float A_BUF[I][K];
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=block factor=NUM_TILES_I dim=1
    #pragma HLS ARRAY_PARTITION variable=A_BUF type=complete dim=2

    float B_BUF[K][J];
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_BUF type=block factor=NUM_TILES_J dim=2

    loadInputsFromDRAM(A_DRAM, B_DRAM, A_BUF, B_BUF);

}

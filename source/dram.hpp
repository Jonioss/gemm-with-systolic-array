#include "constants.h"

void load_A(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, hls::stream<hls::vector<float, K>> A_in[NUM_TILES_I]) {
    #pragma HLS INLINE off
    A_DRAM.read_request(0, I*K/VEC_SIZE);
    for(int ti = 0; ti < NUM_TILES_I; ti++) {
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS PIPELINE II=2
            #pragma HLS LOOP_FLATTEN
            hls::vector<float, K> temp_A_in;
            for(int k = 0; k < K/VEC_SIZE; k++) {
                hls::vector<float, VEC_SIZE> temp_A_DRAM = A_DRAM.read();
                for(int v = 0; v < VEC_SIZE; v++) {
                    #pragma HLS UNROLL
                    temp_A_in[k*VEC_SIZE + v] = temp_A_DRAM[v];
                }
            }
            A_in[ti].write(temp_A_in);
        }
    }
}

void load_B(hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, float B_BUF[K][J]) {
	#pragma HLS INLINE off
	B_DRAM.read_request(0, K*J/VEC_SIZE);
	load_B:
	for(int k = 0; k < K/VEC_SIZE; k++) {
        #pragma HLS PIPELINE II=1
		#pragma HLS LOOP_FLATTEN
		for(int j = 0; j < J; j++) {
			const hls::vector<float, VEC_SIZE> b_vec = B_DRAM.read();
			for(int v = 0; v < VEC_SIZE; v++) {
				#pragma HLS UNROLL
				B_BUF[k*VEC_SIZE+v][j] = b_vec[v];
			}
		}
	}
}

void loadInputsFromDRAM(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM,
    hls::stream<hls::vector<float, K>> A_in[NUM_TILES_I], float B_BUF[K][J]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    load_A(A_DRAM, A_in);
    load_B(B_DRAM, B_BUF);
}

void storeOutputToDRAM(hls::stream<hls::vector<float, S_A_J>> C_out[NUM_TILES_I], hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM) {
    #pragma HLS INLINE off
    C_DRAM.write_request(0, I*J/VEC_SIZE);
    for (int ti = 0; ti < NUM_TILES_I; ti++) {
        #pragma HLS PIPELINE II=S_A_I
        #pragma HLS LOOP_FLATTEN
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS UNROLL
            hls::vector<float, S_A_J> C_out_vec_J = C_out[ti].read();
            C_DRAM.write(C_out_vec_J);
        }
    }
    C_DRAM.write_response();
}


#include "constants.h"

void load_A(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, 
    hls::stream<hls::vector<float, K>> A_in[NUM_TILES_I][NUM_TILES_J]) {
    #pragma HLS INLINE off
    A_DRAM.read_request(0, I*K/VEC_SIZE);
    for(int ti = 0; ti < NUM_TILES_I; ti++) {
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS PIPELINE II=(K/VEC_SIZE)
            #pragma HLS LOOP_FLATTEN
            hls::vector<float, K> temp_A_in;
            for(int k = 0; k < K/VEC_SIZE; k++) {
                hls::vector<float, VEC_SIZE> temp_A_DRAM = A_DRAM.read();
                for(int v = 0; v < VEC_SIZE; v++) {
                    #pragma HLS UNROLL
                    temp_A_in[k*VEC_SIZE + v] = temp_A_DRAM[v];
                }
            }
            for(int tj = 0; tj < NUM_TILES_J; tj++) {
                #pragma HLS UNROLL
                A_in[ti][tj].write(temp_A_in);
            }
        }
    }
}

void load_B(hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, hls::stream<hls::vector<float, K>> B_in[NUM_TILES_J]) {
	#pragma HLS INLINE off
	B_DRAM.read_request(0, K*J/VEC_SIZE);
	load_B:
    for(int tj = 0; tj < NUM_TILES_J; tj++) {
        for(int k = 0; k < K/VEC_SIZE; k++) {
            for(int j = 0; j < S_A_J; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                const hls::vector<float, VEC_SIZE> b_vec = B_DRAM.read();
                hls::vector<float, K> temp_B_in_vec;
                for(int v = 0; v < VEC_SIZE; v++) {
                    #pragma HLS UNROLL
                    temp_B_in_vec[k*VEC_SIZE+v] = b_vec[v];
                }
                B_in[tj].write(temp_B_in_vec);
            }
        }
    }
}

void loadInputsFromDRAM(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM,
    hls::stream<hls::vector<float, K>> A_in[NUM_TILES_I][NUM_TILES_J], hls::stream<hls::vector<float, K>> B_in[NUM_TILES_J]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    load_A(A_DRAM, A_in);
    load_B(B_DRAM, B_in);
}

void storeOutputToDRAM(hls::stream<hls::vector<float, S_A_J>> C_out[NUM_TILES_I][NUM_TILES_J], 
    hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM) {
    #pragma HLS INLINE off
    C_DRAM.write_request(0, I*J/VEC_SIZE);
    for (int ti = 0; ti < NUM_TILES_I; ti++) {
        for(int i = 0; i < S_A_I; i++) {
            #pragma HLS PIPELINE II=(NUM_TILES_J*S_A_J/VEC_SIZE)
            #pragma HLS LOOP_FLATTEN
            for(int tj = 0; tj < NUM_TILES_J; tj++) {
                hls::vector<float, S_A_J> C_out_vec_J = C_out[ti][tj].read();
                for (int jv = 0; jv < S_A_J / VEC_SIZE; jv++) {
                    #pragma HLS UNROLL
                    hls::vector<float, VEC_SIZE> temp;
                    for (int v = 0; v < VEC_SIZE; v++) {
                        #pragma HLS UNROLL
                        temp[v] = C_out_vec_J[jv*VEC_SIZE + v];
                    }
                    C_DRAM.write(temp);
                }
            }
        }
    }
    C_DRAM.write_response();
}


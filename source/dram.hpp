#include "constants.h"

void load_A(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, float A_BUF[I][K]) {
    #pragma HLS INLINE off
    A_DRAM.read_request(0, I*K/VEC_SIZE);
    for(int i = 0; i < I; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int k = 0; k < K/VEC_SIZE; k++) {
            hls::vector<float, VEC_SIZE> temp = A_DRAM.read();
            for(int v = 0; v < VEC_SIZE; v++) {
                #pragma HLS UNROLL
                A_BUF[i][k*VEC_SIZE + v] = temp[v];
            }
        }
    }
}

// Function to read B's values from DRAM to its local buffer
void load_B(hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, float B_BUF[NUM_TILES_J][K][J/NUM_TILES_J]) {
	#pragma HLS INLINE off // Pragma to disable inlining for this function
	B_DRAM.read_request(0, K*J/VEC_SIZE); // Request to DRAM to read B's values
	load_B:
	for(int k = 0; k < K/VEC_SIZE; k++) {
		for(int t=0; t<NUM_TILES_J; t++){
			for(int j = 0; j < J/NUM_TILES_J; j++) {
				#pragma HLS PIPELINE II=1 // Enable pipelining
				#pragma HLS LOOP_FLATTEN // Flatten the loop hierarchy for better performance
				const hls::vector<float, VEC_SIZE> b_vec = B_DRAM.read(); // Helper vector to read B's values from DRAM, 16 at a single burst
				for(int v = 0; v < VEC_SIZE; v++) {
					#pragma HLS UNROLL // Unroll the loop to allow parallel execution of iterations
					B_BUF[t][k*VEC_SIZE+v][j] = b_vec[v]; // Move B's values to the Buffer
				}
			}
		}
	}
}

void loadInputsFromDRAM(hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM,
     float A_BUF[I][K], float B_BUF[NUM_TILES_J][K][J/NUM_TILES_J]) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    load_A(A_DRAM, A_BUF);
    load_B(B_DRAM, B_BUF);
}

void storeOutputToDRAM(float C_BUF[I][J], hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM) {
    #pragma HLS INLINE off
    C_DRAM.write_request(0, I*J/VEC_SIZE);
    for(int i = 0; i < I; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_FLATTEN
        for(int j = 0; j < J/VEC_SIZE; j++) {
            #pragma HLS UNROLL
            hls::vector<float, VEC_SIZE> temp;
            for(int v = 0; v < VEC_SIZE; v++) {
                #pragma HLS UNROLL
                temp[v] = C_BUF[i][j*VEC_SIZE + v];
            }
            C_DRAM.write(temp);
        }
    }
    C_DRAM.write_response();
}


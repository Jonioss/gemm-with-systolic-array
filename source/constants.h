#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include "hls_burst_maxi.h"
#include "math.h"

#define I 32
#define J 8
#define K 8

#define S_A_I 8
#define S_A_J 8

#define NUM_TILES_I (I / S_A_I)
#define NUM_TILES_J (J / S_A_J)

#define VEC_SIZE 8

void gemm(const hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM);

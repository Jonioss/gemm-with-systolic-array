#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include "hls_burst_maxi.h"
#include <hls_task.h>
#include "math.h"
#include "assert.h"

#define I 128
#define J 128
#define K 32

#define S_A_I 16
#define S_A_J 16

#define NUM_TILES_I (I / S_A_I)
#define NUM_TILES_J (J / S_A_J)

#define VEC_SIZE 16
#define BURST_LENGTH (4096 / 32)

void gemm(const hls::burst_maxi<hls::vector<float, VEC_SIZE>> A_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE>> B_DRAM, 
    const hls::burst_maxi<hls::vector<float, VEC_SIZE>> C_DRAM);

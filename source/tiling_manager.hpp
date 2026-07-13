#include "constants.h"

void load_tile_A(float A_BUF[I][K], float A_TILE[S_A_I][K]) {
    #pragma HLS INLINE off


}

void tm_A(float A_BUF[I][K], hls::stream<float> &A_stream) {
    #pragma HLS INLINE off

    float A_TILE[2][S_A_I][K];
    #pragma HLS STREAM type=pipo variable=A_TILE depth=2

    // load A to A_TILE
    // send A_TILE to A_stream to be then fed into the systolic array
}
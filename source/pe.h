#include "constants.h"

void PE(hls::stream<float> &inA_stream, hls::stream<float> &inB_stream, 
        hls::stream<float> &outA_stream, hls::stream<float> &outB_stream, 
        hls::stream<float> &Cout_stream) {
            #pragma HLS INLINE off
            float acc = 0.0f;
            for(int k = 0; k < K; k++) {
                #pragma HLS PIPELINE II=1 rewind
                float a = inA_stream.read();
                float b = inB_stream.read();
                outA_stream.write(a);
                outB_stream.write(b);
                acc += a * b;
            }
            Cout_stream.write(acc);
}

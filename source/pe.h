#include "constants.h"

class PE {
    
public:
    PE(): acc(0) {}

    void calc(hls::stream<float> &inA_stream, hls::stream<float> &inB_stream, 
        hls::stream<float> &outA_stream, hls::stream<float> &outB_stream) {
        #pragma HLS INLINE off

        float a = inA_stream.read();
        float b = inB_stream.read();
        acc += a * b;
        outA_stream.write(a);
        outB_stream.write(b);
    }

    void run(hls::stream<float> &inA_stream, hls::stream<float> &inB_stream, 
        hls::stream<float> &outA_stream, hls::stream<float> &outB_stream, 
        hls::stream<float> &Cout_stream) {
            #pragma HLS INLINE off
            for(int k = 0; k < K; k++) {
                #pragma HLS PIPELINE II=1 rewind
                calc(inA_stream, inB_stream, outA_stream, outB_stream);
            }
            Cout_stream.write(acc);
        }

    float return_result() {
        #pragma HLS INLINE off
        return acc;
    }

    void clear_acc() {
        #pragma HLS INLINE off
        acc = 0;
    }

private:
    float acc;

};

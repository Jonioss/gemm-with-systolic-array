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
            for(int k = 0; k < K; k++) {
                calc(inA_stream, inB_stream, outA_stream, outB_stream);
            }
            Cout_stream.write(acc);
        }

    float return_result() {
        return acc;
    }

    void clear_acc() {
        acc = 0;
    }

private:
    float acc;

};

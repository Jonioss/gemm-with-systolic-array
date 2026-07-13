#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

#include "constants.h"

using namespace std;

float A_float[I][K];
float B_float[K][J];
float C_float[I][J];

float A[I][K];
float B[K][J];
float C[I][J] = {0};

hls::vector<float, VEC_SIZE> A_vec[I][K/VEC_SIZE];
hls::vector<float, VEC_SIZE> B_vec[K/VEC_SIZE][J];
hls::vector<float, VEC_SIZE> C_vec[I][J/VEC_SIZE];

void generateMats() {

	//Generate A and cast to float
	for(int i = 0; i < I; i++) {
		for(int k = 0; k < K; k++) {
			A_float[i][k] = static_cast<float>(rand()) / RAND_MAX;
			A[i][k] = (float) A_float[i][k];
		}
	}

	//Generate B and cast to float
	for(int k = 0; k < K; k++) {
		for(int j = 0; j < J; j++) {
			B_float[k][j] = static_cast<float>(rand()) / RAND_MAX;
			B[k][j] = (float) B_float[k][j];
		}
	}

	//Generate golden C using float
	for(int i = 0; i < I; i++) {
		for(int j = 0; j < J; j++) {
			C_float[i][j] = 0.0f;
			C[i][j] = 0;
			for(int k = 0; k < K; k++) {
				C_float[i][j] += A_float[i][k] * B_float[k][j];
			}
		}
	}
}

void initVectors() {
	for(int i = 0; i < I; i++) {
		for(int k = 0; k < K/VEC_SIZE; k++) {
			for(int v = 0; v < VEC_SIZE; v++) {
				A_vec[i][k][v] = A[i][k*VEC_SIZE + v];
			}
		}
	}
	for(int k = 0; k < K/VEC_SIZE; k++) {
		for(int j = 0; j < J; j++) {
			for(int v = 0; v < VEC_SIZE; v++) {
				B_vec[k][j][v] = B[k*VEC_SIZE+v][j];
			}
		}
	}
	for(int i = 0; i < I; i++) {
		for(int j = 0; j < J/VEC_SIZE; j++) {
			for(int v = 0; v < VEC_SIZE; v++) {
				C_vec[i][j][v] = (float) 0.0;
			}
		}
	}
}

int main() {
	long double MSE = 0.0;

	srand(0);

	generateMats();
	initVectors();
	
	gemm(&A_vec[0][0], &B_vec[0][0], &C_vec[0][0]);

	// Unpack results
	for(int i = 0; i < I; i++) {
		for(int j = 0; j < J/VEC_SIZE; j++) {
			for(int v = 0; v < VEC_SIZE; v++) {
				C[i][j*VEC_SIZE + v] = C_vec[i][j][v];
			}
		}
	}

	for(int i = 0; i < I; i++) {
		for(int j = 0; j < J; j++) {
			MSE += pow((C_float[i][j] - (float)C[i][j]), 2);
		}
	}
	MSE = MSE / (I*J);

	cout << "MSE: " << MSE << endl;

	return 0;
}
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>
#include "transpose.h"
#include "tensor_util.h"
#include "cudacheck.h"
#include "skinny.h"

template<typename T>
void _213transpose(_213TensorUtil<T>& _213tu) {
	size_t vol = _213tu.vol;
	T* h_data = (T*)malloc(vol * sizeof(T));
	_213tu.init_data(h_data);
	
	_213tu.print_mat(h_data);
	
	cudaEvent_t start, stop;
	CudaSafeCall( cudaEventCreate(&start) );
	CudaSafeCall( cudaEventCreate(&stop) );
	CudaSafeCall( cudaEventRecord(start, 0) );
	
	T* d_data = NULL;
	size_t dataSize = vol * sizeof(T);
	CudaSafeCall( cudaMalloc(&d_data, dataSize) );
	CudaSafeCall( cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice) );
	
	int d1 = _213tu.dim[0];
	int d2 = _213tu.dim[1];
	int d3 = _213tu.dim[2];
	for (int k = 0; k < d3; k++) {
		inplace::transpose(true, d_data + k * d1 * d2, d2, d1);
	}
	CudaSafeCall( cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaDeviceSynchronize() );
	CudaSafeCall( cudaEventRecord(stop, 0) );
	CudaSafeCall( cudaEventSynchronize(stop) );
	float t;
	CudaSafeCall( cudaEventElapsedTime(&t, start, stop) );
	//printf("Time: %.5fms\n", t);
	
	_213tu.print_mat(h_data);
	
	CudaSafeCall( cudaFree(d_data) );
	free(h_data);
}

int main(int argc, char** argv) {
	int dim[3];
	dim[0] = atoi(argv[1]);
	dim[1] = atoi(argv[2]);
	dim[2] = atoi(argv[3]);

	int type_size = atoi(argv[4]);
	FILE* fp = (argc == 6)? fopen(argv[5], "wb") : stdout;
	size_t vol = (size_t)dim[0] * dim[1] * dim[2];
	if (type_size == 4) {
		_213TensorUtil<int> _213tu(fp, dim, vol, sizeof(dim));
		_213transpose<int>(_213tu);
	}
	else {
		_213TensorUtil<long long> _213tu(fp, dim, vol, sizeof(dim));
		_213transpose<long long>(_213tu);
	}
	
	return 0;
}
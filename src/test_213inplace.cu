#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda.h>
#include "transpose.h"
#include "tensor_util.h"
#include "cudacheck.h"

template<typename T>
void _213transpose(TensorUtil<T>& tu) {
	size_t& vol = tu.vol;
	//T* h_data = (T*)malloc(vol * sizeof(T));
	T* d_data = NULL;
	size_t dataSize = vol * sizeof(T);
	CudaSafeCall( cudaMallocManaged(&d_data, dataSize) );
	tu.init_data(d_data);
	
	cudaEvent_t start, stop;
	CudaSafeCall( cudaEventCreate(&start) );
	CudaSafeCall( cudaEventCreate(&stop) );
	CudaSafeCall( cudaEventRecord(start, 0) );
	
	//CudaSafeCall( cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice) );
	
	int d1 = tu.dim[0];
	int d2 = tu.dim[1];
	int d3 = tu.dim[2];
	/*for (int k = 0; k < d3; k++) {
		inplace::transpose(true, d_data + k * d1 * d2, d2, d1);
	}*/
	inplace::transpose(d_data, d1, d2, d3);
	//CudaSafeCall( cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaDeviceSynchronize() );
	CudaSafeCall( cudaEventRecord(stop, 0) );
	CudaSafeCall( cudaEventSynchronize(stop) );
	float t;
	CudaSafeCall( cudaEventElapsedTime(&t, start, stop) );
	printf("Time: %.5fms\n", t);
	
	tu.write_file(d_data);
	
	CudaSafeCall( cudaFree(d_data) );
}

int main(int argc, char** argv) {
	int dim[3];
	dim[0] = atoi(argv[1]);
	dim[1] = atoi(argv[2]);
	dim[2] = atoi(argv[3]);

	int type_size = atoi(argv[4]);
	FILE* fp = (argc == 6)? fopen(argv[5], "wb") : stdout;
	int permutation[3] = {1, 0, 2};
	if (type_size == 4) {
		TensorUtil<int> tu(fp, 3, dim, permutation);
		_213transpose<int>(tu);
	}
	else {
		TensorUtil<long long> tu(fp, 3, dim, permutation);
		_213transpose<long long>(tu);
	}
	if (fp != stdout) fclose(fp);
	
	return 0;
}

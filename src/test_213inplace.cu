#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda.h>
#include "transpose.h"
#include "tensor_util.h"
#include "cudacheck.h"

template<typename T>
void verify(T* d_data, size_t dataSize, TensorUtil<T>& tu) {
    T* i_data;
    T* o_data;
    i_data = (T*)malloc(dataSize);
    o_data = (T*)malloc(dataSize);
    tu.init_data(i_data);
    tu.seq_tt(o_data, i_data);
    
    if (memcmp(o_data, d_data, dataSize)) printf("Error\n");
    
    free(i_data);
    free(o_data);
}

template<typename T>
void _213transpose(TensorUtil<T>& tu) {
	size_t& vol = tu.vol;
	//T* h_data = (T*)malloc(vol * sizeof(T));
	T* d_data = NULL;
	size_t dataSize = vol * sizeof(T);
    printf("Data size = %zu Bytes\n", dataSize);
	CudaSafeCall( cudaMallocManaged(&d_data, dataSize) );
	tu.init_data(d_data);
	
	//tu.write_file(d_data);
	
	cudaEvent_t start, stop;
	CudaSafeCall( cudaEventCreate(&start) );
	CudaSafeCall( cudaEventCreate(&stop) );
	CudaSafeCall( cudaEventRecord(start, 0) );
	
	//CudaSafeCall( cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice) );
	
	int d1 = tu.dim[0];
	int d2 = tu.dim[1];
	int d3 = tu.dim[2];
	inplace::transpose(d_data, d1, d2, d3);
    
	//CudaSafeCall( cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaDeviceSynchronize() );
	CudaSafeCall( cudaEventRecord(stop, 0) );
	CudaSafeCall( cudaEventSynchronize(stop) );
	float t;
	CudaSafeCall( cudaEventElapsedTime(&t, start, stop) );
	printf("Time: %.5fms\n", t);
    FILE* txtfp = fopen("time.txt", "a+");
    fprintf(txtfp, "%.5f\n", t);
    fclose(txtfp);
	
    if (tu.fp != NULL) tu.write_file(d_data);
	//tu.print_mat(d_data);
	
	CudaSafeCall( cudaFree(d_data) );
}

int main(int argc, char** argv) {
	int dim[3];
	dim[0] = atoi(argv[1]);
	dim[1] = atoi(argv[2]);
	dim[2] = atoi(argv[3]);

	int type_size = atoi(argv[4]);
	FILE* fp = (argc == 6)? fopen(argv[5], "wb") : NULL;
	int permutation[3] = {1, 0, 2};
	if (type_size == 4) {
		TensorUtil<int> tu(fp, 3, dim, permutation);
		_213transpose<int>(tu);
	}
	else {
		TensorUtil<long long> tu(fp, 3, dim, permutation);
		_213transpose<long long>(tu);
	}
	if (fp != NULL) fclose(fp);
	
	return 0;
}

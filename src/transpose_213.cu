#include <cstdio>
#include <algorithm>
#include "transpose.h"
#include "skinny_213.h"
#include "cudacheck.h"

namespace inplace {
namespace _213 {

template <typename T>
void transpose_fn(T* h_data, int d1, int d2, int d3) {
	//int numStream = d3;
	int numStream = std::min(d3, 512);
	
	cudaStream_t* streams = new cudaStream_t[numStream];
	for (int i = 0; i < numStream; i++) {
		cudaStreamCreate(&streams[i]);
	}
	
	T* d_data = NULL;
	T** temp = new T*[numStream];
	size_t vol = (long long)d1 * d2 * d3;
	CudaSafeCall( cudaMalloc(&d_data, vol * sizeof(T)) );
	for (int i = 0; i < numStream; i++) {
		CudaSafeCall( cudaMalloc(&temp[i], std::max(d1, d2) * sizeof(T)) );
	}
	size_t chunkSize = (long long)d1 * d2 * sizeof(T);
	for (int k = 0; k < d3; k++) {
		long long offset = (long long)k * d1 * d2;
		int sid = k % numStream;
		CudaSafeCall( cudaMemcpyAsync(d_data + offset, h_data + offset, chunkSize, cudaMemcpyHostToDevice, streams[sid]) );
		_2d::transpose(streams[sid], true, (float*)(d_data + offset), (float*)temp[sid], d2, d1);
		CudaSafeCall( cudaMemcpyAsync(h_data + offset, d_data + offset, chunkSize, cudaMemcpyDeviceToHost, streams[sid]) );
	}
	CudaSafeCall( cudaDeviceSynchronize() );
	
	CudaSafeCall( cudaFree(d_data) );
	for (int i = 0; i < numStream; i++) {
		CudaSafeCall( cudaFree(temp[i]) );
		CudaSafeCall( cudaStreamDestroy(streams[i]) );
	}
}

template <typename T>
void skinny_transpose(T* h_data, int d1, int d2, int d3, bool small_m, bool small_n) {
	size_t value;
	CudaSafeCall( cudaDeviceGetLimit(&value, cudaLimitStackSize) );
    CudaSafeCall( cudaDeviceSetLimit(cudaLimitStackSize, value * 32) );
	
	T* d_data = NULL;
	size_t vol = (long long)d1 * d2 * d3;
	CudaSafeCall( cudaMalloc(&d_data, vol * sizeof(T)) );
	
	int m = d2;
	int n = d1;
	if (!small_m && small_n) {
        std::swap(m, n);
    }
	int sid = 0;
	int numStream = (small_n)? std::min(d3, 128) : min(d3, m);
	cudaStream_t* streams = new cudaStream_t[numStream];
	T** temp = new T*[numStream];
	for (int i = 0; i < numStream; i++) {
		cudaStreamCreate(&streams[i]);
		if (!small_n) {
			CudaSafeCall( cudaMalloc(&temp[i], sizeof(T) * n * m) );
		}
		else {
			temp[i] = NULL;
		}
	}
	int h = (small_n)? d3 / (1024 * 2) : min(d3, m);
	for (int k = 0; k < d3; k += h) {
		long long offset = (long long)k * d1 * d2;
		if (h < d3 && d3 - k < h) {
			h = d3 - k;
		}
		size_t chunkSize = (long long)h * d1 * d2 * sizeof(T);
		CudaSafeCall( cudaMemcpyAsync(d_data + offset, h_data + offset, chunkSize, cudaMemcpyHostToDevice, streams[sid]) );
		if (!small_m && small_n) {
			r2c::skinny_transpose(streams[sid], d_data + offset, temp[sid], n, m, h);
		}
		else {
			c2r::skinny_transpose(streams[sid], d_data + offset, temp[sid], n, m, h);
		}
		CudaSafeCall( cudaMemcpyAsync(h_data + offset, d_data + offset, chunkSize, cudaMemcpyDeviceToHost, streams[sid]) );
		sid = (sid + 1) % numStream;
	}
	CudaSafeCall( cudaDeviceSynchronize() );
	CudaSafeCall( cudaFree(d_data) );
	for (int i = 0; i < numStream; i++) {
		CudaSafeCall( cudaStreamDestroy(streams[i]) );
	}
}

template <typename T>
void transpose(T* data, int d1, int d2, int d3) {
	bool small_m = (d2 <= 32);
    bool small_n = (d1 <= 32);
	
	cudaEvent_t start, stop;
	CudaSafeCall( cudaEventCreate(&start) );
	CudaSafeCall( cudaEventCreate(&stop) );
	CudaSafeCall( cudaEventRecord(start, 0) );
	
	if (small_m || small_n) {
		//printf("skinny_transpose\n");
		skinny_transpose(data, d1, d2, d3, small_m, small_n);
	}
	else {
		//printf("transpose_fn\n");
		transpose_fn(data, d1, d2, d3);
	}
	
	CudaSafeCall( cudaEventRecord(stop, 0) );
	CudaSafeCall( cudaEventSynchronize(stop) );
	float t;
	CudaSafeCall( cudaEventElapsedTime(&t, start, stop) );
	printf("Time: %.5fms\n", t);
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}
}
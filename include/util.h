#pragma once

#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include "introspect.h"
#include "cudacheck.h"
#include "chrono"

inline void check_error(std::string message="") {
    cudaError_t error = cudaGetLastError();
    if(error) {
        throw thrust::system_error(error, thrust::cuda_category(), message);
    }
}


__host__ __device__ __forceinline__
unsigned int div_up(unsigned int a, unsigned int b) {
    return (a-1)/b + 1;
}

__host__ __device__ __forceinline__
unsigned int div_down(unsigned int a, unsigned int b) {
    return a / b;
}

__device__
inline size_t chunk_left(size_t id, size_t p, size_t n) {
	return (id * n) / p;
}

__device__
inline size_t chunk_right(size_t id, size_t p, size_t n) {
	return ((id + 1) * n) / p;
}

template<typename F>
int get_num_block(F func, int n_threads, size_t smem_size) {
    int numBlocksPerSm;
    CudaSafeCall( cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, func, n_threads, smem_size) );
    return numBlocksPerSm * inplace::n_sms();
}

int get_num_thread(int d1);

class Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_tp;
    clock::time_point stop_tp;
    
public:
    void start();
    void stop();
    double elapsed_time();
};

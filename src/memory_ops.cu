#include "memory_ops.h"
#include "index.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void memory_row_shuffle(int m, int n, T* d, T* tmp, F s) {
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            tmp[rm(blockIdx.x, j)] = d[rm(i, s(j))];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[rm(i, j)] = tmp[rm(blockIdx.x, j)];
        }
        __syncthreads();
    }        
}

template __global__ void memory_row_shuffle(int m, int n, float* d, float* tmp, c2r::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, double* d, double* tmp, c2r::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, int* d, int* tmp, c2r::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, long long* d, long long* tmp, c2r::shuffle s);

template __global__ void memory_row_shuffle(int m, int n, float* d, float* tmp, r2c::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, double* d, double* tmp, r2c::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, int* d, int* tmp, r2c::shuffle s);
template __global__ void memory_row_shuffle(int m, int n, long long* d, long long* tmp, r2c::shuffle s);

}
}

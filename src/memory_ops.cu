#include "equations.h"
#include "index.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void memory_row_shuffle(int d3, int d2, int d1, T* d, T* tmp, F s) {
    row_major_index rm(d2, d1);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    size_t d2d3 = (size_t)d2 * (size_t)d3;
    
    size_t tmp_offset = blockIdx.x * d1;
    for (size_t ik = blockIdx.x; ik < d2d3; ik += gridDim.x) {
        size_t k = ik / d3;
        size_t offset = k * d1d2;
        int i = ik % d2;
        s.set_i(i);
        int j = threadIdx.x;
        __syncthreads();
        for(; j < d1; j+= blockDim.x) {
            tmp[tmp_offset + j] = d[offset + rm(i, s(j))];
        }
        __syncthreads();
        j = threadIdx.x;
        for(; j < d1; j+= blockDim.x) {
            d[offset + rm(i, j)] = tmp[tmp_offset + j];
        }
    }
}

template __global__ void memory_row_shuffle(int, int, int, float*, float*, c2r::shuffle);
template __global__ void memory_row_shuffle(int, int, int, double*, double*, c2r::shuffle);

template __global__ void memory_row_shuffle(int, int, int, int*, int*, c2r::shuffle);
template __global__ void memory_row_shuffle(int, int, int, long long*, long long*, c2r::shuffle);

template __global__ void memory_row_shuffle(int, int, int, float*, float*, r2c::shuffle);
template __global__ void memory_row_shuffle(int, int, int, double*, double*, r2c::shuffle);

template __global__ void memory_row_shuffle(int, int, int, int*, int*, r2c::shuffle);
template __global__ void memory_row_shuffle(int, int, int, long long*, long long*, r2c::shuffle);

/*template<typename T, typename F>
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
template __global__ void memory_row_shuffle(int m, int n, long long* d, long long* tmp, r2c::shuffle s);*/

}
}

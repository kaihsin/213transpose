#include "index.h"
#include "smem.h"
#include "smem_ops.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void smem_row_shuffle(int m, int n, T* d, F s) {
    T* shared_row = shared_memory<T>();
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            shared_row[j] = d[rm(i, j)];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[rm(i, j)] = shared_row[s(j)];
        }
        __syncthreads();
    }        
}

template __global__ void smem_row_shuffle(int, int, float*, c2r::shuffle);
template __global__ void smem_row_shuffle(int, int, double*, c2r::shuffle);

template __global__ void smem_row_shuffle(int, int, int*, c2r::shuffle);
template __global__ void smem_row_shuffle(int, int, long long*, c2r::shuffle);

template __global__ void smem_row_shuffle(int, int, float*, r2c::shuffle);
template __global__ void smem_row_shuffle(int, int, double*, r2c::shuffle);

template __global__ void smem_row_shuffle(int, int, int*, r2c::shuffle);
template __global__ void smem_row_shuffle(int, int, long long*, r2c::shuffle);

}
}

#include "index.h"
#include "smem.h"
#include "equations.h"
#include "util.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void small_d1d2_shuffle(int d3, int d2, int d1, T* d, F s) {
	T* smem = shared_memory<T>();
	row_major_index rm(d2, d1);
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t k = blockIdx.x; k < d3; k += gridDim.x) {
		size_t offset = k * d1d2;
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			smem[idx] = d[offset + idx];
		}
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t i = (idx / d1);
			s.set_i(i);
			d[offset + idx] = smem[rm(i, s(j))];
		}
	}
}

template __global__ void small_d1d2_shuffle(int, int, int, float*, c2r::shuffle);
template __global__ void small_d1d2_shuffle(int, int, int, double*, c2r::shuffle);
template __global__ void small_d1d2_shuffle(int, int, int, int*, c2r::shuffle);
template __global__ void small_d1d2_shuffle(int, int, int, long long*, c2r::shuffle);
template __global__ void small_d1d2_shuffle(int, int, int, float*, r2c::shuffle);
template __global__ void small_d1d2_shuffle(int, int, int, double*, r2c::shuffle);
template __global__ void small_d1d2_shuffle(int, int, int, int*, r2c::shuffle);
template __global__ void small_d1d2_shuffle(int, int, int, long long*, r2c::shuffle);

template<typename T, typename F>
__global__ void compress_row_shuffle(int d3, int d2, int d1, size_t smem_size, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t d2d3 = (size_t)d2 * (size_t)d3;
    size_t l = chunk_left(blockIdx.x, gridDim.x, d2d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2d3);
    size_t batch_size = smem_size / (size_t)d1;
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1;
        size_t idx = threadIdx.x;
        __syncthreads();
        for (; idx < batch_size * d1; idx += blockDim.x) {
            smem[idx] = d[offset + idx];
        }
        
        idx = threadIdx.x;
        __syncthreads();
        for (; idx < batch_size * d1; idx += blockDim.x) {
            int u = (idx / d1);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
            s.set_i(i);
            d[offset + idx] = smem[u * d1 + s(j)];
        }
    }
}

template __global__ void compress_row_shuffle(int, int, int, size_t, float*, c2r::shuffle);
template __global__ void compress_row_shuffle(int, int, int, size_t, double*, c2r::shuffle);
template __global__ void compress_row_shuffle(int, int, int, size_t, int*, c2r::shuffle);
template __global__ void compress_row_shuffle(int, int, int, size_t, long long*, c2r::shuffle);
template __global__ void compress_row_shuffle(int, int, int, size_t, float*, r2c::shuffle);
template __global__ void compress_row_shuffle(int, int, int, size_t, double*, r2c::shuffle);
template __global__ void compress_row_shuffle(int, int, int, size_t, int*, r2c::shuffle);
template __global__ void compress_row_shuffle(int, int, int, size_t, long long*, r2c::shuffle);

template<typename T, typename F>
__global__ void smem_row_shuffle(int d3, int d2, int d1, T* d, F s) {
    T* shared_row = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    size_t d2d3 = (size_t)d2 * (size_t)d3;
    
    for (size_t ik = blockIdx.x; ik < d2d3; ik += gridDim.x) {
        size_t i = ik % d2;
        size_t k = ik / d2;
        size_t offset = k * d1d2;
        s.set_i(i);
        int j = threadIdx.x;
        __syncthreads();
        for(; j < d1; j+= blockDim.x) {
            shared_row[j] = d[offset + rm(i, s(j))];
        }
        j = threadIdx.x;
        __syncthreads();
        for(;j < d1; j+= blockDim.x) {
            d[offset + rm(i, j)] = shared_row[j];
        }
    }
}

template __global__ void smem_row_shuffle(int, int, int, float*, c2r::shuffle);
template __global__ void smem_row_shuffle(int, int, int, double*, c2r::shuffle);
template __global__ void smem_row_shuffle(int, int, int, int*, c2r::shuffle);
template __global__ void smem_row_shuffle(int, int, int, long long*, c2r::shuffle);
template __global__ void smem_row_shuffle(int, int, int, float*, r2c::shuffle);
template __global__ void smem_row_shuffle(int, int, int, double*, r2c::shuffle);
template __global__ void smem_row_shuffle(int, int, int, int*, r2c::shuffle);
template __global__ void smem_row_shuffle(int, int, int, long long*, r2c::shuffle);

/*template<typename T, typename F>
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
template __global__ void smem_row_shuffle(int, int, long long*, r2c::shuffle);*/

}
}

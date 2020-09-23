#include "skinny.h"
#include "util.h"
#include "equations.h"
#include "smem.h"

namespace inplace {

namespace detail {

template<typename F, typename T>
__global__ void smem_col_shuffle(F fn, int d3, int d2, int d1, T* data) {
	T* smem = shared_memory<T>();
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t k = 0; k < d3; k++) {
		size_t offset = k * d1d2;
		for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				smem[i * blockDim.x + threadIdx.x] = data[offset + i * d1 + j];
			}
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				size_t i_prime = fn(i, j);
				data[offset + i * d1 + j] = smem[i_prime * blockDim.x + threadIdx.x];
			}
		}
	}
}

template<typename F, typename T>
void col_shuffle_fn(F fn, int d3, int d2, int d1, T* data) {
	//printf("col_shuffle_fn\n");
	size_t smem_lim = shared_mem_per_block();
	int x_lim = smem_lim / (d2 * sizeof(T));
	int n_threads_x = min(1024, (int)pow(2, (int)log2(x_lim)));
	int n_threads_y = min(d2, 1024 / n_threads_x);
	dim3 block_dim(n_threads_x, n_threads_y);
	int n_threads = n_threads_x * n_threads_y;
	size_t smem_size = sizeof(T) * d2 * n_threads_x;
	int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_shuffle<F, T>, n_threads, smem_size));
	smem_col_shuffle<<<n_blocks, block_dim, smem_size>>>(fn, d3, d2, d1, data);
}

template void col_shuffle_fn(c2r::fused_postop, int, int, int, float*);
template void col_shuffle_fn(c2r::fused_postop, int, int, int, double*);
template void col_shuffle_fn(c2r::fused_postop, int, int, int, int*);
template void col_shuffle_fn(c2r::fused_postop, int, int, int, long long*);

template void col_shuffle_fn(r2c::fused_preop, int, int, int, float*);
template void col_shuffle_fn(r2c::fused_preop, int, int, int, double*);
template void col_shuffle_fn(r2c::fused_preop, int, int, int, int*);
template void col_shuffle_fn(r2c::fused_preop, int, int, int, long long*);
}
}
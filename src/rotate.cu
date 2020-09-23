#include "rotate.h"
#include "util.h"
#include "equations.h"
#include "smem.h"
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>

namespace inplace {
namespace detail {

__device__ __forceinline__
unsigned int ctz(unsigned int x) {
    return __ffs(x) - 1;
}

__device__ __forceinline__
unsigned int gcd(unsigned int x, unsigned int y) {
    if (x == 0) return y;
    if (y == 0) return x;
    unsigned int cf2 = ctz(x | y);
    x >>= ctz(x);
    while (true) {
        y >>= ctz(y);
        if (x == y) break;
        if (x > y) {
            unsigned int t = x; x = y; y = t;
        }
        if (x == 1) break;
        y -= x;
    }
    return x << cf2;
}

template<typename F, typename T>
__global__ void fine_col_rotate(F fn, int d3, int d2, int d1, T* d) {
    __shared__ T smem[32 * 32]; 

    //If the whole warp is rotating by 0, early exit
    
    //size_t l = chunk_left(blockIdx.x, gridDim.x, d3);
    //size_t r = chunk_right(blockIdx.x, gridDim.x, d3);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    int warp_id = threadIdx.x & 0x1f;
    for (int k = blockIdx.x; k < d3; k += gridDim.x) {
    //for (int k = l; k < r; k++) {
        size_t offset = k * d1d2;
        for (int col = threadIdx.x + blockIdx.y * blockDim.x; col < d1; col += gridDim.y * blockDim.x) {
            //int col = threadIdx.x + blockIdx.x * blockDim.x;
            int coarse_rotation_amount = fn(fn.master(col, warp_id, 32));
            int overall_rotation_amount = fn(col);
            int fine_rotation_amount = overall_rotation_amount - coarse_rotation_amount;
            if (fine_rotation_amount < 0) fine_rotation_amount += d2;
            unsigned warp_vote = __ballot_sync(0xffffffff, fine_rotation_amount > 0);
            if (warp_vote > 0) {
                int row = threadIdx.y;
                int idx = row * d1 + col;
                T* read_ptr = d + offset + idx;
            
                int smem_idx = threadIdx.y * 32 + threadIdx.x;

                T first = -2;
                if (row < d2) first = *read_ptr;

                bool first_phase = (threadIdx.y >= fine_rotation_amount);
                int smem_row = threadIdx.y - fine_rotation_amount;
                if (!first_phase) smem_row += 32;

                int smem_write_idx = smem_row * 32 + threadIdx.x;

                if (first_phase) smem[smem_write_idx] = first;

                T* write_ptr = read_ptr;
                int ptr_inc = 32 * d1;
                read_ptr += ptr_inc;
                //Loop over blocks that are guaranteed not to fall off the edge
                for(int i = 0; i < (d2 / 32) - 1; i++) {
                    T tmp = *read_ptr;
                    if (!first_phase) smem[smem_write_idx] = tmp;
                    __syncthreads();
                    *write_ptr = smem[smem_idx];
                    __syncthreads();
                    if (first_phase) smem[smem_write_idx] = tmp;
                    write_ptr = read_ptr;
                    read_ptr += ptr_inc;
                }

                //Final block (read_ptr may have fallen off the edge)
                int remainder = d2 % 32;
                T tmp = -3;
                if (threadIdx.y < remainder) tmp = *read_ptr;
                int tmp_dest_row = 32 - fine_rotation_amount + threadIdx.y;
                if ((tmp_dest_row >= 0) && (tmp_dest_row < 32))
                    smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
                __syncthreads();
                int first_dest_row = 32 + remainder - fine_rotation_amount + threadIdx.y;
                if ((first_dest_row >= 0) && (first_dest_row < 32))
                    smem[first_dest_row * 32 + threadIdx.x] = first;
            
                __syncthreads();
                *write_ptr = smem[smem_idx];
                write_ptr = read_ptr;
                __syncthreads();
                //Final incomplete block
                tmp_dest_row -= 32; first_dest_row -= 32;
                if ((tmp_dest_row >= 0) && (tmp_dest_row < 32))
                    smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
                __syncthreads();
                if ((first_dest_row >= 0) && (first_dest_row < 32))
                    smem[first_dest_row * 32 + threadIdx.x] = first;
                __syncthreads();
                if (threadIdx.y < remainder) *write_ptr = smem[smem_idx];
            }
        }
    }
}

template<typename F, typename T>
__global__ void small_d1d2_rotate(F fn, int d3, int d2, int d1, T* d) {
	T* smem = shared_memory<T>();
	row_major_index rm(d2, d1);
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t k = blockIdx.x; k < d3; k += gridDim.x) {
		size_t offset = k * d1d2;
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t i = ((idx / d1) - fn(j) + d2) % d2;
			smem[rm(i, j)] = d[offset + idx];
		}
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			d[offset + idx] = smem[idx];
		}
	}
}

template<typename T>
__global__ void coarse(int d2, int d1, int start_col, int n_cycles, uint64_t rotation_amount, T* d) {
	size_t col = threadIdx.x + start_col;
	//for (size_t kst = blockIdx.x; kst < n_cycles * d3; kst += gridDim.x) {
	//	size_t k = kst / n_cycles;
	//	size_t offset = k * (size_t)d1 * (size_t)d2;
	//	size_t start_row = kst % n_cycles;
	//for (size_t col = threadIdx.x + start_col; col < d1; col += blockDim.x) {
	size_t k = blockIdx.x / n_cycles;
	size_t offset = k * (size_t)d1 * (size_t)d2;
	size_t start_row = blockIdx.x % n_cycles;
	size_t src = (start_row + threadIdx.y * rotation_amount) % d2;
	T write_tmp = d[offset + src * d1 + col];
	size_t dest = (src + rotation_amount) % d2;
	
	uint64_t inc = blockDim.y * rotation_amount;
	size_t len = (d2 / n_cycles) / blockDim.y;
	for (size_t i = 0; i < len; i++) {
		size_t next = (src + inc) % d2;
		T read_tmp = d[offset + next * d1 + col];
		__syncthreads();
		d[offset + dest * d1 + col] = write_tmp;
		src = next;
		dest = (src + rotation_amount) % d2;
		write_tmp = read_tmp;
	}
	if (threadIdx.y < (d2 / n_cycles) % blockDim.y) {
		d[offset + dest * d1 + col] = write_tmp;
	}
	//}
	//}
}

template<typename T>
__global__ void coarse_long_col(int d3, int d2, int d1, int master, int n_cycles, uint64_t rotation_amount, size_t start_row, T* d) {
	namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

	int col = threadIdx.x + master;
	for (size_t k = 0; k < d3; k++) {
		size_t offset = k * (size_t)d1 * (size_t)d2;
		size_t src = (start_row + (blockIdx.x * blockDim.y + threadIdx.y) * rotation_amount) % d2;
		T write_tmp = d[offset + src * d1 + col];
		size_t dest = (src + rotation_amount) % d2;
		
		uint64_t inc = gridDim.x * blockDim.y * rotation_amount;
		size_t len = (d2 / n_cycles) / (gridDim.x * blockDim.y);
		for (size_t i = 0; i < len; i++) {
			size_t next = (src + inc) % d2;
			T read_tmp = d[offset + next * d1 + col];
			g.sync();
			d[offset + dest * d1 + col] = write_tmp;
			src = next;
			dest = (src + rotation_amount) % d2;
			write_tmp = read_tmp;
		}
		if (threadIdx.y < (d2 / n_cycles) % (gridDim.x * blockDim.y)) {
			d[offset + dest * d1 + col] = write_tmp;
		}
	}
}

template<typename F, typename T>
void cycle_coarse_enact(F fn, int d3, int d2, int d1, T* data) {
	for (int col = 0; col < d1; col += 32) {
		int master = fn.master(col, 0, 32);
		int rotation_amount = d2 - fn(master);
		int n_cycles = (rotation_amount == d2)? 0 : std::__gcd(rotation_amount, d2);
		if (n_cycles <= 0) continue;
		int cycle_len = d2 / n_cycles;
		int block_x = min(32, d1 - col);
		int block_y = min(cycle_len, low_bit(1024 / block_x));
		dim3 block_dim(block_x, block_y);
		int n_blocks = d3 * n_cycles;
		
		coarse<<<n_blocks, block_dim>>>(d2, d1, col, n_cycles, rotation_amount, data);
	}
}

template<typename F, typename T>
__global__ void smem_col_rotate(F fn, int d3, int d2, int d1, T* data) {
	T* smem = shared_memory<T>();
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t k = 0; k < d3; k++) {
		size_t offset = k * d1d2;
		for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				int i_prim = (i - fn(j) + d2) % d2;
				smem[i_prim * blockDim.x + threadIdx.x] = data[offset + i * d1 + j];
			}
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				data[offset + i * d1 + j] = smem[i * blockDim.x + threadIdx.x];
			}
		}
	}
}

template<typename F, typename T>
void rotate(F fn, int d3, int d2, int d1, T* data) {
	size_t smem_lim = shared_mem_per_block();
	size_t smem_size;
	if ((smem_size = d1 * d2 * sizeof(T)) <= smem_lim) {
		//printf("small_d1d2_rotate\n");
		int n_threads = 1024;
		int n_blocks = min(d3, get_num_block(small_d1d2_rotate<F, T>, n_threads, smem_size));
		small_d1d2_rotate<<<n_blocks, n_threads, smem_size>>>(fn, d3, d2, d1, data);
	}
	else if (smem_lim / (d2 * sizeof(T)) >= 32) {
		//printf("smem_col_rotate\n");
		int x_lim = smem_lim / (d2 * sizeof(T));
		int n_threads_x = min(1024, (int)pow(2, (int)log2(x_lim)));
		int n_threads_y = min(d2, 1024 / n_threads_x);
		dim3 block_dim(n_threads_x, n_threads_y);
		int n_threads = n_threads_x * n_threads_y;
		smem_size = sizeof(T) * d2 * n_threads_x;
		int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_rotate<F, T>, n_threads, smem_size));
		smem_col_rotate<<<n_blocks, block_dim, smem_size>>>(fn, d3, d2, d1, data);
	}
	else {
		if (fn.fine()) {
			dim3 block_dim(32, 32);
			int n_threads = block_dim.x * block_dim.y;
			int n_blocks_x = min(d3, get_num_block(fine_col_rotate<F, T>, n_threads, sizeof(T) * n_threads));
			int n_blocks_y = (n_blocks_x + d3 - 1) / d3;
			dim3 grid_dim(n_blocks_x, n_blocks_y);
			//printf("n_blocks = %d\n", get_num_block(fine_col_rotate<F, T>, n_threads, sizeof(T) * n_threads));
			fine_col_rotate<<<grid_dim, block_dim>>>(fn, d3, d2, d1, data);
		}
		cycle_coarse_enact(fn, d3, d2, d1, data);
	}
}

template void rotate(c2r::prerotator, int, int, int, float*);
template void rotate(c2r::prerotator, int, int, int, double*);
template void rotate(c2r::prerotator, int, int, int, int*);
template void rotate(c2r::prerotator, int, int, int, long long*);

template void rotate(c2r::postrotator, int, int, int, float*);
template void rotate(c2r::postrotator, int, int, int, double*);
template void rotate(c2r::postrotator, int, int, int, int*);
template void rotate(c2r::postrotator, int, int, int, long long*);

template void rotate(r2c::prerotator, int, int, int, float*);
template void rotate(r2c::prerotator, int, int, int, double*);
template void rotate(r2c::prerotator, int, int, int, int*);
template void rotate(r2c::prerotator, int, int, int, long long*);

template void rotate(r2c::postrotator, int, int, int, float*);
template void rotate(r2c::postrotator, int, int, int, double*);
template void rotate(r2c::postrotator, int, int, int, int*);
template void rotate(r2c::postrotator, int, int, int, long long*);


}
}

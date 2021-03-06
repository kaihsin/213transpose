#include <vector>
#include "gcd.h"
#include "index.h"
#include "introspect.h"
#include "util.h"
#include "equations.h"
#include "smem.h"
#include <utility>
#include <algorithm>
#include <queue>
#include <cstdio>

namespace inplace {
namespace detail {

using vector_pair = std::vector<std::pair<int, int> >;
    
template<typename Fn>
void scatter_cycles(Fn f, int* heads, int* lens, vector_pair& vp) {
    std::vector<bool> visited(f.len(), false);
    for (size_t head = 0; head < visited.size(); head++) {
        if (visited[head]) continue;
        visited[head] = true;
        int idx = head;
        int dest = f(idx);
        if (idx != dest) {
            int start = idx;
            int len = 1;
            while (dest != start) {
                idx = dest;
                visited[idx] = true;
                dest = f(idx);
                len++;
            }
            if (len > 1) {
                vp.push_back(std::make_pair(len, head));
            }
        }
    }

    sort(vp.begin(), vp.end());
	reverse(vp.begin(), vp.end());
    
    size_t n_heads = vp.size();
    for (size_t i = 0; i < n_heads; i++) {
		heads[i] = vp[i].second;
		lens[i] = vp[i].first;
    }
}

template<typename T, typename F, int U>
__global__ void cycle_row_permute(F f, T* data, int* heads,
                                  int* lens, int n_heads, int d3) {
    int d1 = f.n;
    int d2 = f.m;
    row_major_index rm(d2, d1);
    size_t d1d2 = (size_t)d1 * (size_t)d2;

	size_t hk = blockIdx.x;
	size_t k = hk / (size_t)n_heads;
	size_t offset = k * d1d2;
	size_t h = hk % n_heads;
	int i = heads[h];
	int l = lens[h];
	for (int j = threadIdx.x; j < d1; j += blockDim.x) {
		T src = data[offset + rm(i, j)];
		T loaded[U+1];
		loaded[0] = src;
		for(int k = 0; k < l / U; k++) {
			int rows[U];
	#pragma unroll
			for(int x = 0; x < U; x++) {
				i = f(i);
				rows[x] = i;
			}
	#pragma unroll
			for(int x = 0; x < U; x++) {
				loaded[x+1] = data[offset + rm(rows[x], j)];
			}
	#pragma unroll
			for(int x = 0; x < U; x++) {
				data[offset + rm(rows[x], j)] = loaded[x];
			}
			loaded[0] = loaded[U];
		}
		T tmp = loaded[0];
		for(int k = 0; k < l % U; k++) {
			i = f(i);
			T new_tmp = data[offset + rm(i, j)];
			data[offset + rm(i, j)] = tmp;
			tmp = new_tmp;
		}
	}
}

template<typename T, typename F>
__global__ void small_d1d2_permute(F f, int d3, int d2, int d1, T* d) {
	T* smem = shared_memory<T>();
	row_major_index rm(d2, d1);
	size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t k = blockIdx.x; k < d3; k += gridDim.x) {
		size_t offset = k * d1d2;
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			size_t j = idx % d1;
			size_t i = f(idx / d1);
			smem[rm(i, j)] = d[offset + idx];
		}
		__syncthreads();
		for (size_t idx = threadIdx.x; idx < d1d2; idx += blockDim.x) {
			d[offset + idx] = smem[idx];
		}
	}
}

template<typename T, typename F>
void scatter_permute(F f, int d3, int d2, int d1, T* data) {
	
	size_t smem_size = d1 * d2 * sizeof(T);
	if (smem_size <= shared_mem_per_block()) {
		int n_threads = 1024;
		int n_blocks = min(d3, get_num_block(small_d1d2_permute<T, F>, n_threads, smem_size));
		small_d1d2_permute<<<n_blocks, n_threads, smem_size>>>(f, d3, d2, d1, data);
	}
	else {
		int* d_heads;
		CudaSafeCall( cudaMallocManaged(&d_heads, sizeof(int) * d2) );
		int* d_lens = d_heads + d2 / 2;
		vector_pair vp;
		scatter_cycles(f, d_heads, d_lens, vp);
		int n_threads = (d1 >= 1024)? 1024 : 64;
		int n_heads = (int)vp.size();
		//int active_blocks = get_num_block(cycle_row_permute<T, F, 4>, n_threads, 0);
		int n_blocks = n_heads * d3; //min(n_heads * d3, active_blocks);
		cycle_row_permute<T, F, 4><<<n_blocks, n_threads>>>(f, data, d_heads, d_lens, n_heads, d3);
	}
}


template void scatter_permute(c2r::scatter_postpermuter, int, int, int, float*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, int, double*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, int, int*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, int, long long*);

template void scatter_permute(r2c::scatter_prepermuter, int, int, int, float*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, int, double*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, int, int*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, int, long long*);


}
}
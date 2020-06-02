#include "skinny_213.h"
#include "skinny.h"
#include "introspect.h"
#include "index.h"
#include "gcd.h"
#include "reduced_math.h"
#include "equations.h"
#include "smem.h"
#include <cassert>

#include <iostream>
#include <cstdio>
#include <cuda.h>
#include "cudacheck.h"

namespace inplace {
namespace _213 {

template<typename T, typename F>
__global__ void register_row_shuffle(int m, int n, T* d, F s) {
	T tmp[32];
	int offset = blockIdx.x * m * n;
	row_major_index rm(m, n);
	int i = threadIdx.y;
    s.set_i(i);
	for (int j = threadIdx.x; j < n; j += blockDim.x) {
		tmp[j] = d[offset + rm(i, s(j))];
	}
	__syncthreads();
	for (int j = threadIdx.x; j < n; j += blockDim.x) {
		d[offset + rm(i, j)] = tmp[j];
	}
}

template<typename T, typename F, int U>
__global__ void long_row_shuffle(int m, int n, int i, T* d, T* tmp, F s) {
    row_major_index rm(m, n);
    s.set_i(i);
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
    int j = global_id;
	int offset = blockIdx.y * n;
    while(j + U * grid_size < n) {
        #pragma unroll
        for(int k = 0; k < U; k++) {
            tmp[offset + j] = d[rm(i, s(j))];
            j += grid_size;
        }
    }
    while(j < n) {
        tmp[offset + j] = d[rm(i, s(j))];
        j += grid_size;
    }
}

template<typename T, typename F>
void skinny_row_op(cudaStream_t& stream, F s, int m, int n, int h, T* d, T* temp) {
	if (n <= 32) {
		dim3 grid_dim(h, 1);
		dim3 block_dim(32, m);
		register_row_shuffle<T, F><<<grid_dim, block_dim, 0, stream>>>(m, n, d, s);
	}
	else {
		printf("long row shuffle\n");
		dim3 grid_dim((n-1)/(256*4)+1, m);
		for(int i = 0; i < h; i++) {
			long_row_shuffle<T, F, 4><<<grid_dim, 256, 0, stream>>>(m, n, i, d, temp, s);
			cudaMemcpyAsync(d, temp, sizeof(T) * n * m, cudaMemcpyDeviceToDevice, stream);
		}
	}
}

template<typename T, typename F>
__global__ void short_column_permute(int m, int n, T* d, F s) {
    T* smem = _2d::shared_memory<T>();
	long long offset = (long long)blockIdx.y * m * n;
    row_major_index rm(m, n);
    row_major_index blk(blockDim.y, blockDim.x);
    int i = threadIdx.y; // One block tall by REQUIREMENT
    int grid_size = blockDim.x * gridDim.x;
    
    if (i < m) {
        for(int j = threadIdx.x + blockIdx.x * blockDim.x;
            j < n; j+= grid_size) {
            
            smem[blk(i, threadIdx.x)] = d[offset + rm(i, j)];
            __syncthreads();
            d[offset + rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
            __syncthreads();

        }   
    }
}

template<typename T, typename F>
void skinny_col_op(cudaStream_t& stream, F s, int m, int n, int h, T* d) {
	//printf("skinny_col_op\n");
    int n_threads = 32;
    // XXX Potential optimization here: figure out how many blocks/sm
    // we should launch
    //int n_blocks = 32;
	if (n <= 32) {
		int n_blocks = 1;
		dim3 grid_dim(n_blocks, h);
		dim3 block_dim(n_threads, m);
		short_column_permute<<<grid_dim, block_dim,
			sizeof(T) * m * n_threads, stream>>>(m, n, d, s);
	}
	else {
		int n_threads = 32;
		int n_blocks = n_sms()*8;
		dim3 grid_dim(n_blocks);
		dim3 block_dim(n_threads, m);
		for (int i = 0; i < h; i++) {
			short_column_permute<<<grid_dim, block_dim,
				sizeof(T) * m * n_threads>>>(m, n, d + m * n, s);
		}
	}
}


namespace c2r {

template<typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int d1, int d2, int d3) {
    //std::cout << "Doing Skinny C2R transpose of " << m << ", " << n << std::endl;

	//printf("Doing Skinny C2R transpose\n");
	int m = d2;
	int n = d1;
    assert(m <= 32);
    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }

	if (c > 1) {
			skinny_col_op(stream, _2d::c2r::fused_preop(m, n/c), m, n, d3, data);
		}
		skinny_row_op(stream, _2d::c2r::long_shuffle(m, n, c, k), m, n, d3, data, temp);
		skinny_col_op(stream, _2d::c2r::fused_postop(m, n, c), m, n, d3, data);
}


template void skinny_transpose(cudaStream_t& stream, float* data, float* temp, int d1, int d2, int d3);
template void skinny_transpose(cudaStream_t& stream, double* data, double* temp, int d1, int d2, int d3);
template void skinny_transpose(cudaStream_t& stream, int* data, int* temp, int d1, int d2, int d3);
template void skinny_transpose(cudaStream_t& stream, long long* data, long long* temp, int d1, int d2, int d3);

}

namespace r2c {

template<typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int d1, int d2, int d3) {
    //std::cout << "Doing Skinny R2C transpose of " << m << ", " << n << std::endl;
	//printf("Doing Skinny R2C transpose\n");

	int m = d2;
	int n = d1;
    assert(m <= 32);
    int c, t, q;
    extended_gcd(n, m, c, t);
    if (c > 1) {
        extended_gcd(n/c, m/c, t, q);
    } else {
        q = t;
    }
	
	skinny_col_op(stream, _2d::r2c::fused_preop(m/c, c, m, q), m, n, d3, data);
	skinny_row_op(stream, _2d::r2c::shuffle(m, n, c, 0), m, n, d3, data, temp);
	if (c > 1) {
		skinny_col_op(stream, _2d::r2c::fused_postop(m, n/c), m, n, d3, data);
	}
}

template void skinny_transpose(cudaStream_t& stream, float* data, float* temp, int d1, int d2, int d3);
template void skinny_transpose(cudaStream_t& stream, double* data, double* temp, int d1, int d2, int d3);
template void skinny_transpose(cudaStream_t& stream, int* data, int* temp, int d1, int d2, int d3);
template void skinny_transpose(cudaStream_t& stream, long long* data, long long* temp, int d1, int d2, int d3);

}


}
}

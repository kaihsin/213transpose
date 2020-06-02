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

#include "save_array.h"

namespace inplace {
namespace _2d {

template<typename T, typename F, int U>
__global__ void long_row_shuffle(int m, int n, int i, T* d, T* tmp, F s) {
    row_major_index rm(m, n);
    s.set_i(i);
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
    int j = global_id;
    while(j + U * grid_size < n) {
        #pragma unroll
        for(int k = 0; k < U; k++) {
            tmp[j] = d[rm(i, s(j))];
            j += grid_size;
        }
    }
    while(j < n) {
        tmp[j] = d[rm(i, s(j))];
        j += grid_size;
    }
}

template<typename T, typename F>
__global__ void short_column_permute(int m, int n, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(m, n);
    row_major_index blk(blockDim.y, blockDim.x);
    int i = threadIdx.y; // One block tall by REQUIREMENT
    int grid_size = blockDim.x * gridDim.x;
    
    if (i < m) {
        for(int j = threadIdx.x + blockIdx.x * blockDim.x;
            j < n; j+= grid_size) {
            
            smem[blk(i, threadIdx.x)] = d[rm(i, j)];
            __syncthreads();
            d[rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
            __syncthreads();

        }   
    }
}

template<typename T, typename F>
__global__ void register_row_shuffle(int m, int n, T* d, F s) {
	__shared__ T tmp[1024];
	row_major_index rm(m, n);
	int i = threadIdx.x;
    s.set_i(i);
	for (int j = threadIdx.y; j < n; j += blockDim.y) {
		tmp[i * n + j] = d[rm(i, s(j))];
	}
	__syncthreads();
	for (int j = threadIdx.y; j < n; j += blockDim.y) {
		d[rm(i, j)] = tmp[i * n + j];
	}
}

template<typename T, typename F>
void skinny_row_op(cudaStream_t& stream, F s, int m, int n, T* d, T* tmp) {
    for(int i = 0; i < m; i++) {
        long_row_shuffle<T, F, 4><<<(n-1)/(256*4)+1,256, 0, stream>>>(m, n, i, d, tmp, s);
        cudaMemcpy(d + n * i, tmp, sizeof(T) * n, cudaMemcpyDeviceToDevice);

    }
}

template<typename T, typename F>
void skinny_col_op(cudaStream_t& stream, F s, int m, int n, T* d) {
    int n_threads = 32;
    // XXX Potential optimization here: figure out how many blocks/sm
    // we should launch
    int n_blocks = n_sms()*8;
    dim3 grid_dim(n_blocks);
    dim3 block_dim(n_threads, m);
    short_column_permute<<<grid_dim, block_dim,
        sizeof(T) * m * n_threads, stream>>>(m, n, d, s);
}


namespace c2r {

template<typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int m, int n) {
    //std::cout << "Doing Skinny C2R transpose of " << m << ", " << n << std::endl;
	//printf("Doing Skinny C2R transpose\n");

    assert(m <= 32);
    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }

    if (c > 1) {
        skinny_col_op(stream, fused_preop(m, n/c), m, n, data);
    }
    //T* tmp;
    //cudaMalloc(&tmp, sizeof(T) * n);
	row_major_index rm(m, n);
	
    skinny_row_op(stream, long_shuffle(m, n, c, k), m, n, data, temp);
    //cudaFree(tmp);
    skinny_col_op(stream, fused_postop(m, n, c), m, n, data);

}


template void skinny_transpose(cudaStream_t& stream, float* data, float* temp, int m, int n);
template void skinny_transpose(cudaStream_t& stream, double* data, double* temp, int m, int n);
template void skinny_transpose(cudaStream_t& stream, int* data, int* temp, int m, int n);
template void skinny_transpose(cudaStream_t& stream, long long* temp, long long* data, int m, int n);

}

namespace r2c {

template<typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int m, int n) {
    //std::cout << "Doing Skinny R2C transpose of " << m << ", " << n << std::endl;
	//printf("Doing Skinny R2C transpose\n");

    assert(m <= 32);
    int c, t, q;
    extended_gcd(n, m, c, t);
    if (c > 1) {
        extended_gcd(n/c, m/c, t, q);
    } else {
        q = t;
    }

    skinny_col_op(stream, fused_preop(m/c, c, m, q), m, n, data);
    //T* tmp;
    //cudaMalloc(&tmp, sizeof(T) * n);
    skinny_row_op(stream, shuffle(m, n, c, 0), m, n, data, temp);
    //cudaFree(tmp);
    if (c > 1) {
        skinny_col_op(stream, fused_postop(m, n/c), m, n, data);
    }
}

template void skinny_transpose(cudaStream_t& stream, float* data, float* temp, int m, int n);
template void skinny_transpose(cudaStream_t& stream, double* data, double* temp, int m, int n);
template void skinny_transpose(cudaStream_t& stream, int* data, int* temp, int m, int n);
template void skinny_transpose(cudaStream_t& stream, long long* temp, long long* data, int m, int n);

}



}
}

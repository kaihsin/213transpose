#include "introspect.h"
#include "index.h"
#include "gcd.h"
#include "reduced_math.h"
#include "equations.h"
#include "smem.h"
#include "skinny.h"
#include <cassert>

#include <iostream>
#include <cstdio>
#include <cooperative_groups.h>

#include "save_array.h"
#include "cudacheck.h"

namespace inplace {
namespace detail {

template<typename T, typename F, int U>
__global__ void long_row_shuffle(int d3, int d2, int d1, T* d, T* tmp, F s) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
    row_major_index rm(d2, d1);
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
    for (int k = 0; k < d3; k++) {
        size_t offset = (size_t)k * (size_t)d1 * (size_t)d2;
        for (int i = 0; i < d2; i++) {
            s.set_i(i);
            int j = global_id;
            /*while(j + U * grid_size < d1) {
                #pragma unroll
                for(int k = 0; k < U; k++) {
                    tmp[j] = d[offset + rm(i, s(j))];
                    j += grid_size;
                }
            }*/
            while(j < d1) {
                tmp[j] = d[offset + rm(i, s(j))];
                j += grid_size;
            }

            g.sync();

            j = global_id;
            /*while(j + U * grid_size < d1) {
                #pragma unroll
                for(int k = 0; k < U; k++) {
                    d[offset + rm(i, j)] = tmp[j];
                    j += grid_size;
                }
            }*/
            while(j < d1) {
                d[offset + rm(i, j)] = tmp[j];
                j += grid_size;
            }
            g.sync();
        }
    }
}

template<typename T, typename F>
__global__ void smem_row_shuffle(int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t offset = blockIdx.x * (size_t)d1 * (size_t)d2;
    for (int i = 0; i < d2; i++) {
        s.set_i(i);
        __syncthreads();
        for (int j = threadIdx.x; j < d1; j += blockDim.x) {
            smem[j] = d[offset + rm(i, j)];
        }
        __syncthreads();
        for (int j = threadIdx.x; j < d1; j += blockDim.x) {
            d[offset + rm(i, j)] = smem[s(j)];
        }
    }
}

/*template<typename T, typename F>
__global__ void short_column_permute(int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    row_major_index blk(blockDim.y, blockDim.x);
    int i = threadIdx.y; // One block tall by REQUIREMENT
    int grid_size = blockDim.x * gridDim.x;
    //size_t offset = blockIdx.y * (size_t)d1 * (size_t)d2;
    
    if (i < d2) {
        for(int j = threadIdx.x + blockIdx.x * blockDim.x;
            j < d1; j+= grid_size) {
            
            smem[blk(i, threadIdx.x)] = d[rm(i, j)];
            __syncthreads();
            d[rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
            __syncthreads();

        }
    }
}*/

template<typename T, typename F>
__global__ void short_column_permute(int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    row_major_index blk(blockDim.y, blockDim.x);
    int i = threadIdx.y; // One block tall by REQUIREMENT
    int grid_size_x = blockDim.x * gridDim.y;
    size_t offset = blockIdx.x * (size_t)d1 * (size_t)d2;
    
    for(int j = threadIdx.x + blockIdx.y * blockDim.x; j < d1; j += grid_size_x) {
        smem[blk(i, threadIdx.x)] = d[offset + rm(i, j)];
        __syncthreads();
        d[offset + rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
        __syncthreads();
    }
}

template<typename T, typename F>
void global_mem_row_op(F s, int d3, int d2, int d1, T* d) {
    T* tmp;
    CudaSafeCall( cudaMalloc(&tmp, sizeof(T) * d1) );
    int n_threads = 1024;
    int numBlocksPerSm;
    CudaSafeCall( cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, long_row_shuffle<T, F, 4>, n_threads, 0) );
    int n_blocks = numBlocksPerSm * n_sms();
    void *kernelArgs[] = {
        (void *)&d3, (void *)&d2, (void *)&d1, (void *)&d, (void *)&tmp, (void *)&s
    };
    CudaSafeCall( cudaLaunchCooperativeKernel((void *)long_row_shuffle<T, F, 4>,
                                          n_blocks, n_threads, kernelArgs) );
    CudaSafeCall( cudaFree(tmp) );
}

template<typename T, typename F>
void skinny_row_op(F s, int d3, int d2, int d1, T* d) {
    size_t smem_size = sizeof(T) * d1;
    cudaDeviceProp prop;
    CudaSafeCall( cudaGetDeviceProperties(&prop, 0) );
    if (prop.sharedMemPerBlock >= smem_size) {
        int n_threads = 32;
        smem_row_shuffle<<<d3, n_threads, smem_size>>>(d2, d1, d, s);
    }
    else {
        global_mem_row_op(s, d3, d2, d1, d);
    }
    
    /*for(int i = 0; i < d2; i++) {
    
        //long_row_shuffle<T, F, 4><<<(d1-1)/(256*4)+1,256>>>(d2, d1, i, d, tmp, s);
        //cudaMemcpy(d + d1 * i, tmp, sizeof(T) * d1, cudaMemcpyDeviceToDevice);

    }*/
}

template<typename T, typename F>
void skinny_col_op(F s, int d3, int d2, int d1, T* d) {
    int n_threads = 32;
    // XXX Potential optimization here: figure out how many blocks/sm
    // we should launch
    size_t smem_size = sizeof(T) * d2 * n_threads;
    int numBlocksPerSm;
    CudaSafeCall( cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, short_column_permute<T, F>, n_threads, smem_size) );
    int n_blocks = numBlocksPerSm * n_sms();
    /*dim3 grid_dim(n_blocks);
    dim3 block_dim(n_threads, d2);
    //Naive For loop  32 32 695800 6733.55225ms; For loop in kernel 2901.36499ms

    for (int i = 0; i < d3; i++) {
        short_column_permute<<<grid_dim, block_dim,
            smem_size>>>(d2, d1, d + i * d1 * d2, s);
    }*/
    
    /*dim3 grid_dim(n_blocks);
    dim3 block_dim(n_threads, d2);
    short_column_permute<<<grid_dim, block_dim,
        smem_size>>>(d3, d2, d1, d, s);*/
    
    dim3 grid_dim(d3, max(1, n_blocks / d3));
    dim3 block_dim(n_threads, d2);
    short_column_permute<<<grid_dim, block_dim,
        smem_size>>>(d2, d1, d, s);
}

namespace c2r {

template<typename T>
void skinny_transpose(T* data, int d1, int d2, int d3) {
    //std::cout << "Doing Skinny C2R transpose of " << d2 << ", " << d1 << std::endl;
    printf("Doing Skinny C2R transpose\n");

    assert(d2 <= 32);

    int c, t, k;
    extended_gcd(d2, d1, c, t);
    if (c > 1) {
        extended_gcd(d2/c, d1/c, t, k);
    } else {
        k = t;
    }
    
    if (c > 1) {
        skinny_col_op(fused_preop(d2, d1/c), d3, d2, d1, data);
    }
    skinny_row_op(long_shuffle(d2, d1, c, k), d3, d2, d1, data);
    skinny_col_op(fused_postop(d2, d1, c), d3, d2, d1, data);
}

template void skinny_transpose(int*, int, int, int);
template void skinny_transpose(long long*, int, int, int);
template void skinny_transpose(float*, int, int, int);
template void skinny_transpose(double*, int, int, int);

}

namespace r2c {

template<typename T>
void skinny_transpose(T* data, int d1, int d2, int d3) {
    //std::cout << "Doing Skinny R2C transpose of " << d2 << ", " << d1 << std::endl;
    //printf("Doing Skinny R2C transpose\d1");

    assert(d2 <= 32);
    int c, t, q;
    extended_gcd(d1, d2, c, t);
    if (c > 1) {
        extended_gcd(d1/c, d2/c, t, q);
    } else {
        q = t;
    }
    
    skinny_col_op(fused_preop(d2/c, c, d2, q), d3, d2, d1, data);
    skinny_row_op(shuffle(d2, d1, c, 0), d3, d2, d1, data);
    if (c > 1) {
        skinny_col_op(fused_postop(d2, d1/c), d3, d2, d1, data);
    }
}

template void skinny_transpose(int*, int, int, int);
template void skinny_transpose(long long*, int, int, int);
template void skinny_transpose(float*, int, int, int);
template void skinny_transpose(double*, int, int, int);

}



}
}

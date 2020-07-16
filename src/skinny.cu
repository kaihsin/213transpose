#include "introspect.h"
#include "index.h"
#include "gcd.h"
#include "reduced_math.h"
#include "equations.h"
#include "smem.h"
#include "skinny.h"
#include "util.h"
#include <cassert>
#include <cmath>

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
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    for (size_t k = 0; k < d3; k++) {
        size_t offset = k * d1d2;
        for (int i = 0; i < d2; i++) {
            s.set_i(i);
            int j = global_id;
            g.sync();
            /*while(j + U * grid_size < d1) {
                #pragma unroll
                for(int u = 0; u < U; u++) {
                    tmp[j] = d[offset + rm(i, s(j))];
                    j += grid_size;
                }
            }*/
            while(j < d1) {
                tmp[j] = d[offset + rm(i, s(j))];
                j += grid_size;
            }

            j = global_id;
            g.sync();
            /*while(j + U * grid_size < d1) {
                #pragma unroll
                for(int u = 0; u < U; u++) {
                    d[offset + rm(i, j)] = tmp[j];
                    j += grid_size;
                }
            }*/
            while(j < d1) {
                d[offset + rm(i, j)] = tmp[j];
                j += grid_size;
            }
        }
    }
}

template<typename T, typename F>
__global__ void smem_row_shuffle(int d3, int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t l = chunk_left(blockIdx.x, gridDim.x, d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d3);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    size_t offset = l * d1d2;
    for (size_t k = l; k < r; k++) {
        for (int i = 0; i < d2; i++) {
            s.set_i(i);
            int j = threadIdx.x;
            __syncthreads();
            for (; j < d1; j += blockDim.x) {
                smem[j] = d[offset + rm(i, j)];
            }
            j = threadIdx.x;
            __syncthreads();
            for (; j < d1; j += blockDim.x) {
                d[offset + rm(i, j)] = smem[s(j)];
            }
        }
        offset += d1d2;
    }
}

template<typename T, typename F>
__global__ void short_row_shuffle(int d3, int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t l = chunk_left(blockIdx.x, gridDim.x, d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d3);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    size_t offset = l * d1d2;
    for (size_t k = l; k < r; k++) {
        size_t idx = threadIdx.x;
        __syncwarp();
        for (; idx < d1d2; idx += blockDim.x) {
            smem[idx] = d[offset + idx];
        }
        idx = threadIdx.x;
        __syncwarp();
        for (; idx < d1d2; idx += blockDim.x) {
            size_t i = idx / d1;
            size_t j = idx % d1;
            s.set_i(i);
            d[offset + idx] = smem[rm(i, s(j))];
        }
        offset += d1d2;
    }
    /*for (size_t k = blockIdx.x; k < d3; k += gridDim.x) {
        size_t offset = k * (size_t)d1 * (size_t)d2;
        __syncwarp();
        for (size_t idx = threadIdx.x; idx < d1 * d2; idx += blockDim.x) {
            smem[idx] = d[offset + idx];
        }
        __syncwarp();
        for (size_t idx = threadIdx.x; idx < d1 * d2; idx += blockDim.x) {
            int i = idx / d1;
            int j = idx % d1;
            s.set_i(i);
            d[offset + idx] = smem[rm(i, s(j))];
        }
    }*/
}

template<typename T, typename F>
__global__ void compress_row_shuffle(int d3, int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t l = chunk_left(blockIdx.x, gridDim.x, d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d3);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    size_t batch_size = 32 / d1d2;
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * d1d2;
        size_t idx = threadIdx.x;
        __syncwarp();
        if (idx < batch_size * d1d2) {
            smem[idx] = d[offset + idx];
        }
        
        size_t k = idx / d1d2;
        size_t i = (idx % d1d2) / d1;
        size_t j = (idx % d1d2) % d1;
        s.set_i(i);
        __syncwarp();
        if (idx < batch_size * d1d2) {
            d[offset + idx] = smem[k * d1d2 + rm(i, s(j))];
        }
    }
}

template<typename T, typename F>
__global__ void short_column_permute(int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    row_major_index blk(blockDim.y, blockDim.x);
    int i = threadIdx.y; // One block tall by REQUIREMENT
    int grid_size_x = blockDim.x * gridDim.y;
    size_t offset = blockIdx.x * (size_t)d1 * (size_t)d2;
    
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < d1;
        j += grid_size_x) {
        __syncthreads();
        smem[blk(i, threadIdx.x)] = d[offset + rm(i, j)];
        __syncthreads();
        d[offset + rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
    }
}

template<typename T, typename F>
__global__ void small_d1d2_permute(int d3, int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t l = chunk_left(blockIdx.x, gridDim.x, d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d3);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    size_t offset = l * d1d2;
    for (size_t k = l; k < r; k++) {
        size_t idx = threadIdx.x;
        __syncthreads();
        for (; idx < d1 * d2; idx += blockDim.x) {
            smem[idx] = d[offset + idx];
        }
        
        idx = threadIdx.x;
        __syncthreads();
        for (; idx < d1 * d2; idx += blockDim.x) {
            size_t i = idx / d1;
            size_t j = idx % d1;
            d[offset + idx] = smem[rm(s(i, j), j)];
        }
        offset += d1d2;
    }
}

template<typename T, typename F>
__global__ void compress_column_permute(int d3, int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    size_t l = chunk_left(blockIdx.x, gridDim.x, d3);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d3);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    size_t batch_size = 32 / d1d2;
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * d1d2;
        size_t idx = threadIdx.x;
        __syncwarp();
        if (idx < batch_size * d1d2) {
            smem[idx] = d[offset + idx];
        }
        
        size_t k = idx / d1d2;
        size_t i = (idx % d1d2) / d1;
        size_t j = (idx % d1d2) % d1;
        __syncwarp();
        if (idx < batch_size * d1d2) {
            d[offset + idx] = smem[k * d1d2 + rm(s(i, j), j)];
        }
    }
}

template<typename F>
int get_num_block(F func, int n_threads, size_t smem_size) {
    int numBlocksPerSm;
    CudaSafeCall( cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, func, n_threads, smem_size) );
    return numBlocksPerSm * n_sms();
}

template<typename T, typename F>
void skinny_row_op(F s, int d3, int d2, int d1, T* d) {
    if (d1 * d2 <= 32) {
        size_t smem_size = sizeof(T) * 32;
        int n_threads = 32;
        int n_blocks = get_num_block(compress_row_shuffle<T, F>, n_threads, smem_size);
        n_blocks = min(n_blocks, d3);
        compress_row_shuffle<<<n_blocks, n_threads, smem_size>>>(d3, d2, d1, d, s);
    }
    else if (shared_mem_per_block() / 8 >= sizeof(T) * d1 * d2) {
        //printf("short_row_shuffle\n");
        size_t smem_size = sizeof(T) * d1 * d2;
        int n_threads = 32;
        int n_blocks = get_num_block(short_row_shuffle<T, F>, n_threads, smem_size);
        n_blocks = min(n_blocks, d3);
        short_row_shuffle<<<n_blocks, n_threads, smem_size>>>(d3, d2, d1, d, s);
    }
    else if (shared_mem_per_block() >= sizeof(T) * d1) {
        size_t smem_size = sizeof(T) * d1;
        //printf("smem_row_shuffle\n");
        int n_threads = 256;
        int n_blocks = get_num_block(smem_row_shuffle<T, F>, n_threads, smem_size);
        n_blocks = min(n_blocks, d3);
        smem_row_shuffle<<<n_blocks, n_threads, smem_size>>>(d3, d2, d1, d, s);
    }
    else {
        T* tmp;
        CudaSafeCall( cudaMalloc(&tmp, sizeof(T) * d1) );
        int n_threads = 1024;
        int n_blocks = get_num_block(long_row_shuffle<T, F, 4>, n_threads, 0);
        void *kernelArgs[] = {
            (void *)&d3, (void *)&d2, (void *)&d1, (void *)&d, (void *)&tmp, (void *)&s
        };
        CudaSafeCall( cudaLaunchCooperativeKernel((void *)long_row_shuffle<T, F, 4>,
                                              n_blocks, n_threads, kernelArgs) );
        CudaSafeCall( cudaFree(tmp) );
    }
    
    /*for(int i = 0; i < d2; i++) {
    
        //long_row_shuffle<T, F, 4><<<(d1-1)/(256*4)+1,256>>>(d2, d1, i, d, tmp, s);
        //cudaMemcpy(d + d1 * i, tmp, sizeof(T) * d1, cudaMemcpyDeviceToDevice);

    }*/
}

int get_num_thread(int d1) {
    /*int n_threads = 1;
    while (n_threads < d1 && n_threads < 1024) {
        n_threads <<= 1;
    }
    return n_threads;*/
    int msb = static_cast<int>(log2(d1)); // most significant bit
    unsigned n_threads = static_cast<unsigned>(pow(2, msb + 1));
    unsigned lim = 1024;
    return static_cast<int>(min(n_threads, lim));
}

template<typename T, typename F>
void skinny_col_op(F s, int d3, int d2, int d1, T* d) {
    if (d1 * d2 <= 32) {
        size_t smem_size = sizeof(T) * 32;
        int n_threads = 32;
        int n_blocks = get_num_block(compress_column_permute<T, F>, n_threads, smem_size);
        n_blocks = min(n_blocks, d3);
        compress_column_permute<<<n_blocks, n_threads, smem_size>>>(d3, d2, d1, d, s);
    }
    else if (shared_mem_per_block() >= sizeof(T) * d1 * d2) {
        int n_threads = get_num_thread(d1);
        size_t smem_size = sizeof(T) * d2 * d1;
        int n_blocks = get_num_block(small_d1d2_permute<T, F>, n_threads, smem_size);
        n_blocks = min(n_blocks, d3);
        small_d1d2_permute<<<n_blocks, n_threads,
            smem_size>>>(d3, d2, d1, d, s);
    }
    else {
        int n_threads = 32;
        size_t smem_size = sizeof(T) * d2 * n_threads;
        int n_blocks = get_num_block(short_column_permute<T, F>, n_threads * d2, smem_size);

        dim3 grid_dim(d3, (n_blocks + d3 - 1) / d3);
        dim3 block_dim(n_threads, d2);
        short_column_permute<<<grid_dim, block_dim,
            smem_size>>>(d2, d1, d, s);
    }
}

namespace c2r {

template<typename T>
void skinny_transpose(T* data, int d1, int d2, int d3) {
    //std::cout << "Doing Skinny C2R transpose of " << d2 << ", " << d1 << std::endl;
    //printf("Doing Skinny C2R transpose\n");

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

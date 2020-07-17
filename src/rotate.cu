#include "rotate.h"
#include "util.h"
#include "equations.h"

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
__global__ void coarse_col_rotate(F fn, reduced_divisor d2, int d1, T* d) {
    int warp_id = threadIdx.x & 0x1f;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rotation_amount = fn(fn.master(global_index, warp_id, 32));
    int col = global_index;

    __shared__ T smem[32 * 16];
    
    if ((col < d1) && (rotation_amount > 0)) {
        row_major_index rm(d2, d1);
        int c = gcd(rotation_amount, d2.get());
        int l = d2.get() / c;
        size_t inc = d2.get() - rotation_amount;
        int smem_write_idx = threadIdx.y * 32 + threadIdx.x;
        int max_col = (l > 16) ? 15 : l - 1;
        int smem_read_col = (threadIdx.y == 0) ? max_col : (threadIdx.y - 1);
        int smem_read_idx = smem_read_col * 32 + threadIdx.x;
        
        for(int b = 0; b < c; b++) {
            size_t x = threadIdx.y;
            size_t pos = ((size_t)b + x * inc) % (size_t)d2.get();   // (b + x * inc) % d2
            //int pos = d2.mod(b + x * inc);   // (b + x * inc) % d2
            smem[smem_write_idx] = d[rm(pos, col)];
            __syncthreads();
            T prior = smem[smem_read_idx];
            if (x < l) d[rm(pos, col)] = prior;
            __syncthreads();
            int n_rounds = l / 16;
            for(int i = 1; i < n_rounds; i++) {
                x += blockDim.y;
                size_t pos = ((size_t)b + x * inc) % (size_t)d2.get();   // (b + x * inc) % d2
                //int pos = d2.mod(b + x * inc);            
                if (x < l) smem[smem_write_idx] = d[rm(pos, col)];
                __syncthreads();
                T incoming = smem[smem_read_idx];
                T outgoing = (threadIdx.y == 0) ? prior : incoming;
                if (x < l) d[rm(pos, col)] = outgoing;
                prior = incoming;
                __syncthreads();
            }
            //Last round/cleanup
            x += blockDim.y;
            pos = ((size_t)b + x * inc) % (size_t)d2.get();
            //pos = d2.mod(b + x * inc);
            if (x <= l) smem[smem_write_idx] = d[rm(pos, col)];
            __syncthreads();
            int remainder_length = (l % 16);
            int fin_smem_read_col = (threadIdx.y == 0) ? remainder_length : threadIdx.y - 1;
            int fin_smem_read_idx = fin_smem_read_col * 32 + threadIdx.x;
            T incoming = smem[fin_smem_read_idx];
            T outgoing = (threadIdx.y == 0) ? prior : incoming;
            if (x <= l) d[rm(pos, col)] = outgoing;
            
        }
    }
}



template<typename F, typename T>
__global__ void fine_col_rotate(F fn, int d2, int d1, T* d) {
    __shared__ T smem[32 * 32]; 

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = threadIdx.x & 0x1f;
    int coarse_rotation_amount = fn(fn.master(col, warp_id, 32));
    int overall_rotation_amount = fn(col);
    int fine_rotation_amount = overall_rotation_amount - coarse_rotation_amount;
    if (fine_rotation_amount < 0) fine_rotation_amount += d2;
    //If the whole warp is rotating by 0, early exit
    unsigned warp_vote = __ballot_sync(0xffffffff, fine_rotation_amount > 0);
    if (col < d1) {
        if (warp_vote > 0) {
            int row = threadIdx.y;
            int idx = row * d1 + col;
            T* read_ptr = d + idx;
        
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

template<typename F, typename T>
void rotate(F fn, int d3, int d2, int d1, T* data) {
    int n_blocks = div_up(d1, 32);
    
    size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t i = 0; i < d3; i++) {
        size_t offset = i * d1d2;
        if (fn.fine()) {
            dim3 block_dim(32, 32);
            fine_col_rotate<<<n_blocks, block_dim>>>(fn, d2, d1, data + offset);
        }
        coarse_col_rotate<<<n_blocks, dim3(32, 16)>>>(
            fn, d2, d1, data + offset);
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

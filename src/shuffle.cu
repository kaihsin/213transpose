#include "introspect.h"
#include "equations.h"
#include "util.h"
#include "memory_ops.h"
#include "smem_ops.h"
#include "register_ops.h"

namespace inplace {

namespace detail {

template<typename T, typename F>
void enact(T* data, int d3, int d2, int d1, F s) {
	size_t smem_bytes = sizeof(T) * (size_t)d1;
	if (2 * d1 * sizeof(T) <= shared_mem_per_block() / 32) {
        //printf("compress row shuffle\n");
        int n_threads = get_num_thread(d1);
		//int n_threads = 32;
        //printf("n_threads = %d\n", n_threads);
        int n_blocks = min(d2 * d3, get_num_block(compress_row_shuffle<T, F>, n_threads, shared_mem_per_block() / 32));
        //printf("n_blocks = %d\n", n_blocks);
        compress_row_shuffle<<<n_blocks, n_threads, shared_mem_per_block() / 32>>>(d3, d2, d1, shared_mem_per_block() / 32 / sizeof(T), data, s);
    }
    else if (smem_bytes <= shared_mem_per_block()) {
        int n_threads = get_num_thread(d1);
        //printf("n_threads = %d\n", n_threads);
        int n_blocks = min(d2 * d3, get_num_block(smem_row_shuffle<T, F>, n_threads, smem_bytes));
        //printf("n_blocks = %d\n", n_blocks);
        smem_row_shuffle<<<n_blocks, n_threads, smem_bytes>>>(d3, d2, d1, data, s);
        check_error("smem shuffle");
    } else if (sizeof(T) == 4 && d1 < 30720) {
        size_t d1d2 = (size_t)d1 * (size_t)d2;
        for (size_t i = 0; i < d3; i++) {
            size_t offset = i * d1d2;
            register_row_shuffle<T, F, 60>
                <<<d2, 512>>>(d2, d1, data + offset, s);
            check_error("register 60 shuffle");
        }
        
    } else if (sizeof(T) == 8 && d1 < 6918) {
        register_row_shuffle<T, F, 18>
            <<<d2, 512>>>(d2, d1, data, s);
        check_error("register 18 shuffle");
        
    } else if (sizeof(T) == 8 && d1 < 29696) {
        register_row_shuffle<T, F, 57>
            <<<d2, 512>>>(d2, d1, data, s);
        check_error("register 58 shuffle");
        
    } else {
        //printf("memory shuffle\n");
        T* tmp;
        CudaSafeCall( cudaMalloc(&tmp, sizeof(T) * d1) );
        int n_threads = 1024;
        int n_blocks = get_num_block(memory_row_shuffle<T, F>, n_threads, 0);
        void *kernelArgs[] = {
            (void *)&d3, (void *)&d2, (void *)&d1, (void *)&data, (void *)&tmp, (void *)&s
        };
        CudaSafeCall( cudaLaunchCooperativeKernel((void *)memory_row_shuffle<T, F>,
                                              n_blocks, n_threads, kernelArgs) );
        CudaSafeCall( cudaFree(tmp) );
        
    }
}

template<typename T, typename F>
void shuffle_fn(T* data, int d3, int d2, int d1, F s) {
    int arch = current_sm();
    
    if (arch >= 601) {
        enact(data, d3, d2, d1, s);
    }
    else {
        throw std::invalid_argument("Requires sm_61 or greater");
    }
}

template void shuffle_fn(int*, int, int, int, detail::c2r::shuffle);
template void shuffle_fn(int*, int, int, int, detail::r2c::shuffle);
template void shuffle_fn(float*, int, int, int, detail::c2r::shuffle);
template void shuffle_fn(float*, int, int, int, detail::r2c::shuffle);
template void shuffle_fn(long long*, int, int, int, detail::c2r::shuffle);
template void shuffle_fn(long long*, int, int, int, detail::r2c::shuffle);
template void shuffle_fn(double*, int, int, int, detail::c2r::shuffle);
template void shuffle_fn(double*, int, int, int, detail::r2c::shuffle);

}

}

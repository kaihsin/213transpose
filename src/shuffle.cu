#include "introspect.h"
#include "equations.h"
#include "util.h"
#include "memory_ops.h"
#include "smem_ops.h"
#include "register_ops.h"

/*template<typename T, typename F>
void enact(T* data, int d3, int d2, int d1, F s) {
    size_t smem_bytes = sizeof(T) * (size_t)d1;
    /*if (2 * d1 * sizeof(T) <= shared_mem_per_block() / 16) {
        printf("compress row shuffle\n");
        int n_threads = get_num_thread(d1);
        printf("n_threads = %d\n", n_threads);
        int n_blocks = min(d2 * d3, get_num_block(compress_row_shuffle<T, F>, n_threads, shared_mem_per_block() / 16));
        printf("n_blocks = %d\n", n_blocks);
        compress_row_shuffle<<<n_blocks, n_threads, shared_mem_per_block() / 16>>>(d3, d2, d1, shared_mem_per_block() / 16 / sizeof(T), data, s);
    }
    else if (smem_bytes <= shared_mem_per_block()) {
        int n_threads = get_num_thread(d1);
        printf("n_threads = %d\n", n_threads);
        int n_blocks = min(d2 * d3, get_num_block(smem_row_shuffle<T, F>, n_threads, smem_bytes));
        printf("n_blocks = %d\n", n_blocks);
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
        printf("memory shuffle\n");
        int n_threads = get_num_thread(d1) ;
        printf("n_threads = %d\n", n_threads);
        int n_blocks = get_num_block(memory_row_shuffle<T, F>, n_threads, 0);
        printf("n_blocks = %d\n", n_blocks);
        T* temp;
        cudaMalloc(&temp, sizeof(T) * d1 * n_blocks);
        memory_row_shuffle<<<n_blocks, n_threads>>>(d3, d2, d1, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
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
}*/

namespace inplace {

namespace detail {

template<typename T, typename F>
void enact(T* data, int d3, int d2, int d1, F s) {
    printf("Shuffle function (d1, d2, d3) = (%d, %d, %d)\n", d1, d2, d3);
    size_t smem_bytes = sizeof(T) * (size_t)d1;
    /*if (2 * d1 * sizeof(T) <= shared_mem_per_block() / 16) {
        printf("compress row shuffle\n");
        int n_threads = get_num_thread(d1);
        printf("n_threads = %d\n", n_threads);
        int n_blocks = min(d2 * d3, get_num_block(compress_row_shuffle<T, F>, n_threads, shared_mem_per_block() / 16));
        printf("n_blocks = %d\n", n_blocks);
        compress_row_shuffle<<<n_blocks, n_threads, shared_mem_per_block() / 16>>>(d3, d2, d1, shared_mem_per_block() / 16 / sizeof(T), data, s);
    }
    else*/ if (smem_bytes <= shared_mem_per_block()) {
        int n_threads = get_num_thread(d1);
        printf("n_threads = %d\n", n_threads);
        int n_blocks = min(d2 * d3, get_num_block(smem_row_shuffle<T, F>, n_threads, smem_bytes));
        printf("n_blocks = %d\n", n_blocks);
        smem_row_shuffle<<<n_blocks, n_threads, smem_bytes>>>(d3, d2, d1, data, s);
        check_error("smem shuffle");
    } /*else if (sizeof(T) == 4 && d1 < 30720) {
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
        
    }*/ else {
        printf("memory shuffle\n");
        int n_threads = get_num_thread(d1) ;
        printf("n_threads = %d\n", n_threads);
        int n_blocks = get_num_block(memory_row_shuffle<T, F>, n_threads, 0);
        printf("n_blocks = %d\n", n_blocks);
        T* temp;
        cudaMalloc(&temp, sizeof(T) * d1 * n_blocks);
        memory_row_shuffle<<<n_blocks, n_threads>>>(d3, d2, d1, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
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

/*template<typename T, typename F>
void _8byte_enact(T* data, int d2, int d1, F s) {
    if (d1 < 6144) {
        int smem_bytes = sizeof(T) * d1;
        smem_row_shuffle<<<d2, 256, smem_bytes>>>(d2, d1, data, s);
        check_error("smem shuffle");
    } else if (d1 < 6918) {
        register_row_shuffle<T, F, 18>
            <<<d2, 512>>>(d2, d1, data, s);
        check_error("register 18 shuffle");
        
    } else if (d1 < 29696) {
        register_row_shuffle<T, F, 57>
            <<<d2, 512>>>(d2, d1, data, s);
        check_error("register 58 shuffle");
        
    } else {
        T* temp;
        cudaMalloc(&temp, sizeof(T) * d1 * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(d2, d1, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
    }
}

template<typename T, typename F>
void _4byte_enact(T* data, int d2, int d1, F s) {
    if (d1 < 12288) {
        int smem_bytes = sizeof(T) * d1;
        smem_row_shuffle<<<d2, 256, smem_bytes>>>(d2, d1, data, s);
        check_error("smem shuffle");
    } else if (d1 < 30720) {
        register_row_shuffle<T, F, 60>
            <<<d2, 512>>>(d2, d1, data, s);
        check_error("register 60 shuffle");
        
    } else {
        T* temp;
        cudaMalloc(&temp, sizeof(T) * d1 * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(d2, d1, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
    }
}

template<typename T, typename F>
void shuffle_fn(T* data, int d3, int d2, int d1, F s) {
    int arch = current_sm();
	int type_size = sizeof(T);
    
    size_t d1d2 = (size_t)d1 * (size_t)d2;
	for (size_t i = 0; i < d3; i++) {
        size_t offset = i * d1d2;
        if (arch >= 601) {
            if (type_size == 4) _4byte_enact(data + offset, d2, d1, s);
            else _8byte_enact(data + offset, d2, d1, s);
        }
        else {
            throw std::invalid_argument("Requires sm_61 or greater");
        }
    }
}*/

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

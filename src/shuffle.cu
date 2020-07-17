#include "introspect.h"
#include "equations.h"
#include "util.h"
#include "memory_ops.h"
#include "smem_ops.h"
#include "register_ops.h"

namespace inplace {

namespace detail {

template<typename T, typename F>
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

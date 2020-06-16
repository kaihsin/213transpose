#include "introspect.h"
#include "equations.h"
#include "util.h"
#include "memory_ops.h"
#include "smem_ops.h"
#include "register_ops.h"

namespace inplace {

namespace detail {

template<typename T, typename F>
void sm_35_8byte_enact(T* data, int m, int n, F s) {
    if (n < 3072) {
        int smem_bytes = sizeof(T) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 4100) {
        register_row_shuffle<T, F, 16>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 16 shuffle");
        
    } else if (n < 6918) {
        register_row_shuffle<T, F, 18>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 18 shuffle");
        
    } else if (n < 30208) {
        register_row_shuffle<T, F, 59>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        T* temp;
        cudaMalloc(&temp, sizeof(T) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
    }
}

template<typename T, typename F>
void sm_35_4byte_enact(T* data, int m, int n, F s) {
    
    if (n < 6144) {
        int smem_bytes = sizeof(T) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 11326) {
        register_row_shuffle<T, F, 31>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 31 shuffle");
        
    } else if (n < 30720) {
        register_row_shuffle<T, F, 60>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        T* temp;
        cudaMalloc(&temp, sizeof(T) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
    }
}

template<typename T, typename F>
void sm_52_8byte_enact(T* data, int m, int n, F s) {
    if (n < 6144) {
        int smem_bytes = sizeof(T) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 6918) {
        register_row_shuffle<T, F, 18>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 18 shuffle");
        
    } else if (n < 29696) {
        register_row_shuffle<T, F, 57>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 58 shuffle");
        
    } else {
        T* temp;
        cudaMalloc(&temp, sizeof(T) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
    }
}

template<typename T, typename F>
void sm_52_4byte_enact(T* data, int m, int n, F s) {
    if (n < 12288) {
        int smem_bytes = sizeof(T) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 30720) {
        register_row_shuffle<T, F, 60>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        T* temp;
        cudaMalloc(&temp, sizeof(T) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
    }
}

template<typename T, typename F>
void shuffle_fn(T* data, int m, int n, F s) {
    int arch = current_sm();
	int type_size = sizeof(T);
    if (arch >= 502) {
		if (type_size == 4) sm_52_4byte_enact(data, m, n, s);
		else sm_52_8byte_enact(data, m, n, s);
    } else if (arch >= 305) {
        if (type_size == 4) sm_35_4byte_enact(data, m, n, s);
		else sm_35_8byte_enact(data, m, n, s);
    } else {
        throw std::invalid_argument("Requires sm_35 or greater");
    }
}

template void shuffle_fn(int*, int, int, detail::c2r::shuffle);
template void shuffle_fn(int*, int, int, detail::r2c::shuffle);
template void shuffle_fn(float*, int, int, detail::c2r::shuffle);
template void shuffle_fn(float*, int, int, detail::r2c::shuffle);
template void shuffle_fn(long long*, int, int, detail::c2r::shuffle);
template void shuffle_fn(long long*, int, int, detail::r2c::shuffle);
template void shuffle_fn(double*, int, int, detail::c2r::shuffle);
template void shuffle_fn(double*, int, int, detail::r2c::shuffle);

}

}

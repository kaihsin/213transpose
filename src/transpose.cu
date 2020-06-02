#include "gcd.h"
#include "introspect.h"
#include "rotate.h"
#include "permute.h"
#include "equations.h"
#include "skinny.h"
#include "util.h"
#include "memory_ops.h"
#include "smem_ops.h"
#include "register_ops.h"
#include "cudacheck.h"
#include <algorithm>
#include <cstdio>
#include <cuda.h>


namespace inplace {

namespace _2d {

template<typename F>
void sm_35_enact(cudaStream_t& stream, double* data, double* temp, int m, int n, F s) {
    if (n < 3072) {
        int smem_bytes = sizeof(double) * n;
        smem_row_shuffle<<<m, 256, smem_bytes, stream>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 4100) {
        register_row_shuffle<double, F, 16>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 16 shuffle");
        
    } else if (n < 6918) {
        register_row_shuffle<double, F, 18>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 18 shuffle");
        
    } else if (n < 30208) {
        register_row_shuffle<double, F, 59>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        memory_row_shuffle
            <<<n_ctas(), n_threads(), 0, stream>>>(m, n, data, temp, s);
        check_error("memory shuffle");
        
    }
}

template<typename F>
void sm_35_enact(cudaStream_t& stream, float* data, float* temp, int m, int n, F s) {
    if (n < 6144) {
        int smem_bytes = sizeof(float) * n;
        smem_row_shuffle<<<m, 256, smem_bytes, stream>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 11326) {
        register_row_shuffle<float, F, 31>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 31 shuffle");
        
    } else if (n < 30720) {
        register_row_shuffle<float, F, 60>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        memory_row_shuffle
            <<<n_ctas(), n_threads(), 0, stream>>>(m, n, data, temp, s);
        check_error("memory shuffle");
        
    }
}

template<typename F>
void sm_52_enact(cudaStream_t& stream, double* data, double* temp, int m, int n, F s) {
    if (n < 6144) {
        int smem_bytes = sizeof(double) * n;
        smem_row_shuffle<<<m, 256, smem_bytes, stream>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 6918) {
        register_row_shuffle<double, F, 18>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 18 shuffle");
        
    } else if (n < 29696) {
        register_row_shuffle<double, F, 57>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 58 shuffle");
        
    } else {
        memory_row_shuffle
            <<<n_ctas(), n_threads(), 0, stream>>>(m, n, data, temp, s);
        check_error("memory shuffle");
        
    }
}

template<typename F>
void sm_52_enact(cudaStream_t& stream, float* data, float* temp, int m, int n, F s) {
    if (n < 12288) {
        int smem_bytes = sizeof(float) * n;
        smem_row_shuffle<<<m, 256, smem_bytes, stream>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 30720) {
        register_row_shuffle<float, F, 60>
            <<<m, 512, 0, stream>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        memory_row_shuffle
            <<<n_ctas(), n_threads(), 0, stream>>>(m, n, data, temp, s);
        check_error("memory shuffle");
        
    }
}

template<typename T, typename F>
void shuffle_fn(cudaStream_t& stream, T* data, T* temp, int m, int n, F s) {
    int arch = current_sm();
    if (arch >= 502) {
        sm_52_enact(stream, data, temp, m, n, s);
    } else if (arch >= 305) {
        sm_35_enact(stream, data, temp, m, n, s);
    } else {
        throw std::invalid_argument("Requires sm_35 or greater");
    }
}

namespace c2r {

template<typename T>
void transpose_fn(cudaStream_t& stream, bool row_major, T* data, T* temp, int m, int n) {
    if (!row_major) {
        std::swap(m, n);
    }
    //std::cout << "Doing C2R transpose of " << m << ", " << n << std::endl;

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    if (c > 1) {
        _2d::rotate(stream, _2d::c2r::prerotator(n/c), m, n, data);
    }
    _2d::shuffle_fn(stream, data, temp, m, n, _2d::c2r::shuffle(m, n, c, k));
    _2d::rotate(stream, _2d::c2r::postrotator(m), m, n, data);
    _2d::scatter_permute(stream, _2d::c2r::scatter_postpermuter(m, n, c), m, n, data, temp);
}


void transpose(cudaStream_t& stream, bool row_major, float* data, float* temp, int m, int n) {
    transpose_fn(stream, row_major, data, temp, m, n);
}
void transpose(cudaStream_t& stream, bool row_major, double* data, double* temp, int m, int n) {
    transpose_fn(stream, row_major, data, temp, m, n);
}
/*void transpose(bool row_major, int* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}
void transpose(bool row_major, long long* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}*/

}

namespace r2c {

template<typename T>
void transpose_fn(cudaStream_t& stream, bool row_major, T* data, T* temp, int m, int n) {
    if (row_major) {
        std::swap(m, n);
    }
    //std::cout << "Doing R2C transpose of " << m << ", " << n << std::endl;

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }

    _2d::scatter_permute(stream, _2d::r2c::scatter_prepermuter(m, n, c), m, n, data, temp);
    _2d::rotate(stream, _2d::r2c::prerotator(m), m, n, data);
    _2d::shuffle_fn(stream, data, temp, m, n, _2d::r2c::shuffle(m, n, c, k));
    if (c > 1) {
        rotate(stream, _2d::r2c::postrotator(n/c, m), m, n, data);
    }
}


void transpose(cudaStream_t& stream, bool row_major, float* data, float* temp, int m, int n) {
    transpose_fn(stream, row_major, data, temp, m, n);
}
void transpose(cudaStream_t& stream, bool row_major, double* data, double* temp, int m, int n) {
    transpose_fn(stream, row_major, data, temp, m, n);
}

}


template<typename T>
void transpose_fn(cudaStream_t& stream, bool row_major, T* data, T* temp, int m, int n) {
    bool small_m = m < 32;
    bool small_n = n < 32;
    //Heuristic to choose the fastest implementation
    //based on size of matrix and data layout
    if (!small_m && small_n) {
        std::swap(m, n);
        if (!row_major) {
			//printf("c2r::skinny_transpose\n");
            _2d::c2r::skinny_transpose(
                stream, data, temp, m, n);
        } else {
			//printf("r2c::skinny_transpose\n");
            _2d::r2c::skinny_transpose(
                stream, data, temp, m, n);
        }
    } else if (small_m) {
        if (!row_major) {
			//printf("r2c::skinny_transpose\n");
            _2d::r2c::skinny_transpose(
                stream, data, temp, m, n);
        } else {
			//printf("c2r::skinny_transpose\n");
            _2d::c2r::skinny_transpose(
                stream, data, temp, m, n);
        }
    } else {
        bool m_greater = m > n;
        if (m_greater ^ row_major) {
			//printf("r2c::transpose\n");
            _2d::r2c::transpose(stream, row_major, data, temp, m, n);
        } else {
			//printf("c2r::transpose\n");
            _2d::c2r::transpose(stream, row_major, data, temp, m, n);
        }
    }
}

void transpose(cudaStream_t& stream, bool row_major, float* data, float* temp, int m, int n) {
    transpose_fn(stream, row_major, data, temp, m, n);
}
void transpose(cudaStream_t& stream, bool row_major, double* data, double* temp, int m, int n) {
    transpose_fn(stream, row_major, data, temp, m, n);
}
} // end of namespace _2d

}

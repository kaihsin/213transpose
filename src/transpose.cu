#include "gcd.h"
#include "introspect.h"
#include "rotate.h"
#include "permute.h"
#include "equations.h"
#include "skinny.h"
#include "shuffle.h"
#include "col_shuffle.h"
#include "util.h"
#include <algorithm>
#include <cstdio>
#include "cudacheck.h"

namespace inplace {

namespace c2r {

template<typename T>
void col_op(T* data, int d1, int d2, int d3, int c) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 32) {
		detail::col_shuffle_fn(detail::c2r::fused_postop(d2, d1, d2/c), d3, d2, d1, data);
	}
	else {
		detail::rotate(detail::c2r::postrotator(d2), d3, d2, d1, data);
		detail::scatter_permute(detail::c2r::scatter_postpermuter(d2, d1, c), d3, d2, d1, data);
	}
}

template<typename T>
void transpose(T* data, int d1, int d2, int d3) {
    //std::cout << "Doing C2R transpose of " << d2 << ", " << d1 << std::endl;
    printf("Doing C2R transpose\n");
    int c, t, k;
    extended_gcd(d2, d1, c, t);
    if (c > 1) {
        extended_gcd(d2/c, d1/c, t, k);
    } else {
        k = t;
    }

    if (c > 1) {
        detail::rotate(detail::c2r::prerotator(d1/c), d3, d2, d1, data);
    }
    detail::shuffle_fn(data, d3, d2, d1, detail::c2r::shuffle(d2, d1, c, k));
    col_op(data, d1, d2, d3, c);
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}

namespace r2c {

template<typename T>
void col_op(T* data, int d1, int d2, int d3, int c, int q) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 32) {
		detail::col_shuffle_fn(detail::r2c::fused_preop(d2/c, c, d2, q), d3, d2, d1, data);
	}
	else {
		detail::scatter_permute(detail::r2c::scatter_prepermuter(d2, d1, c), d3, d2, d1, data);
		detail::rotate(detail::r2c::prerotator(d2), d3, d2, d1, data);
	}
}

template<typename T>
void transpose(T* data, int d1, int d2, int d3) {
    
    //std::cout << "Doing R2C transpose of " << d2 << ", " << d1 << std::endl;
    printf("Doing R2C transpose\n");

    int c, t, q;
    extended_gcd(d1, d2, c, t);
    if (c > 1) {
        extended_gcd(d1/c, d2/c, t, q);
    } else {
        q = t;
    }
	
	col_op(data, d1, d2, d3, c, q);
    detail::shuffle_fn(data, d3, d2, d1, detail::r2c::shuffle(d2, d1, c, q));
    if (c > 1) {
        detail::rotate(detail::r2c::postrotator(d1/c, d2), d3, d2, d1, data);
    }
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}


template<typename T>
void transpose(T* data, int d1, int d2, int d3) {
	int dev;
    CudaSafeCall( cudaGetDevice(&dev) );
	size_t dataSize = (size_t)d1*(size_t)d2*(size_t)d3*sizeof(T);
    CudaSafeCall( cudaMemPrefetchAsync(data, dataSize, dev, 0) );
    bool small_m = d2 <= 32;
    bool small_n = d1 <= 32;

	if (small_m || small_n) {
		if (d2 < d1) {
			inplace::detail::c2r::skinny_transpose(data, d1, d2, d3);
		}
		else {
            inplace::detail::r2c::skinny_transpose(data, d2, d1, d3);
		}
	}
	else { // For large d1 and d2
		if (/*max(d1, d2) / min(d1, d2) < 1e2 ||*/ d1 >= d2) {
			inplace::c2r::transpose(data, d1, d2, d3);
		}
        else {
            inplace::r2c::transpose(data, d2, d1, d3);
        }
	}
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}

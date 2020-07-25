#include "gcd.h"
#include "introspect.h"
#include "rotate.h"
#include "permute.h"
#include "equations.h"
#include "skinny.h"
#include "shuffle.h"
#include <algorithm>
#include <cstdio>
#include "cudacheck.h"

namespace inplace {

namespace c2r {

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

    int* tmp_int;
    //CudaSafeCall( cudaMalloc(&tmp_int, sizeof(int) * d2) );
    CudaSafeCall( cudaMallocManaged(&tmp_int, sizeof(int) * d2) );
    if (c > 1) {
        detail::rotate(detail::c2r::prerotator(d1/c), d3, d2, d1, data);
    }
    //printf("shuffle\n");
    detail::shuffle_fn(data, d3, d2, d1, detail::c2r::shuffle(d2, d1, c, k));
    //printf("post rotation\n");
    detail::rotate(detail::c2r::postrotator(d2), d3, d2, d1, data);
    //printf("permute\n");
    detail::scatter_permute(detail::c2r::scatter_postpermuter(d2, d1, c), d3, d2, d1, data, tmp_int);
    //printf("done\n");
    
    CudaSafeCall( cudaFree(tmp_int) );
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}

namespace r2c {

template<typename T>
void transpose(T* data, int d1, int d2, int d3) {
    
    //std::cout << "Doing R2C transpose of " << d2 << ", " << d1 << std::endl;
    printf("Doing R2C transpose\n");

    int c, t, k;
    extended_gcd(d2, d1, c, t);
    if (c > 1) {
        extended_gcd(d2/c, d1/c, t, k);
    } else {
        k = t;
    }

    int* tmp_int;
    //CudaSafeCall( cudaMalloc(&tmp_int, sizeof(int) * d2) );
    CudaSafeCall( cudaMallocManaged(&tmp_int, sizeof(int) * d2) );
	
    detail::scatter_permute(detail::r2c::scatter_prepermuter(d2, d1, c), d3, d2, d1, data, tmp_int);
    detail::rotate(detail::r2c::prerotator(d2), d3, d2, d1, data);
    detail::shuffle_fn(data, d3, d2, d1, detail::r2c::shuffle(d2, d1, c, k));
    if (c > 1) {
        detail::rotate(detail::r2c::postrotator(d1/c, d2), d3, d2, d1, data);
    }
    
	CudaSafeCall( cudaFree(tmp_int) );
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}


template<typename T>
void transpose(T* data, int d1, int d2, int d3) {
    bool small_m = d2 <= 32;
    bool small_n = d1 <= 32;

    //Heuristic to choose the fastest implementation
    //based on size of matrix and data layout
	if (small_m || small_n) {
		if (d2 < d1) {
			inplace::detail::c2r::skinny_transpose(data, d1, d2, d3);
		}
		else {
			//std::swap(d1, d2);
			//inplace::detail::r2c::skinny_transpose(data, d1, d2, d3);
            inplace::detail::r2c::skinny_transpose(data, d2, d1, d3);
		}
	}
	else { // For large d1 and d2
        if (d2 < d1) {
			//std::swap(d1, d2);
            //inplace::r2c::transpose(data, d1, d2, d3);
            inplace::r2c::transpose(data, d2, d1, d3);
        }
		else {
            inplace::c2r::transpose(data, d1, d2, d3);
        }
        //inplace::c2r::transpose(data, d1, d2, d3);
	}
	
	
    /*if (!small_m && small_n) {
        std::swap(d2, d1);
        if (!row_major) {
			//fprintf(stdout, "c2r::skinny_transpose\d1");
            inplace::detail::c2r::skinny_transpose(
                data, d2, d1);
        } else {
			//fprintf(stdout, "r2c::skinny_transpose\d1");
            inplace::detail::r2c::skinny_transpose(
                data, d2, d1);
        }
    } else if (small_m) {
        if (!row_major) {
			//fprintf(stdout, "r2c::skinny_transpose\d1");
            inplace::detail::r2c::skinny_transpose(
                data, d2, d1);
        } else {
			//fprintf(stdout, "c2r::skinny_transpose\d1");
            inplace::detail::c2r::skinny_transpose(
                data, d2, d1);
        }
    } else {
        bool m_greater = d2 > d1;
        if (m_greater ^ row_major) {
			//fprintf(stdout, "r2c::transpose\d1");
            inplace::r2c::transpose(row_major, data, d2, d1);
        } else {
			//fprintf(stdout, "c2r::transpose\d1");
            inplace::c2r::transpose(row_major, data, d2, d1);
        }
    }*/
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}

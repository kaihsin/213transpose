#include "gcd.h"
#include "introspect.h"
#include "rotate.h"
#include "permute.h"
#include "equations.h"
#include "skinny.h"
#include "shuffle.h"
#include <algorithm>
#include <cstdio>


namespace inplace {

namespace c2r {

template<typename T>
void transpose(T* data, int d1, int d2, int d3) {
    /*
    //std::cout << "Doing C2R transpose of " << m << ", " << n << std::endl;

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    if (c > 1) {
        detail::rotate(detail::c2r::prerotator(n/c), m, n, data);
    }
    detail::shuffle_fn(data, m, n, detail::c2r::shuffle(m, n, c, k));
    detail::rotate(detail::c2r::postrotator(m), m, n, data);
    int* temp_int;
    cudaMalloc(&temp_int, sizeof(int) * m);
    detail::scatter_permute(detail::c2r::scatter_postpermuter(m, n, c), m, n, data, temp_int);
    cudaFree(temp_int);*/
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}

namespace r2c {

template<typename T>
void transpose(T* data, int d1, int d2, int d3) {
    /*
    //std::cout << "Doing R2C transpose of " << m << ", " << n << std::endl;

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    int* temp_int;
    cudaMalloc(&temp_int, sizeof(int) * m);
    detail::scatter_permute(detail::r2c::scatter_prepermuter(m, n, c), m, n, data, temp_int);
    cudaFree(temp_int);
    detail::rotate(detail::r2c::prerotator(m), m, n, data);
    detail::shuffle_fn(data, m, n, detail::r2c::shuffle(m, n, c, k));
    if (c > 1) {
        detail::rotate(detail::r2c::postrotator(n/c, m), m, n, data);
    }*/
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
			std::swap(d1, d2);
			inplace::detail::r2c::skinny_transpose(data, d1, d2, d3);
		}
	}
	else { // For large d1 and d2
        if (d2 < d1) {
			std::swap(d1, d2);
            inplace::r2c::transpose(data, d1, d2, d3);
        }
		else {
            inplace::c2r::transpose(data, d1, d2, d3);
        }
	}
	
	
    /*if (!small_m && small_n) {
        std::swap(m, n);
        if (!row_major) {
			//fprintf(stdout, "c2r::skinny_transpose\n");
            inplace::detail::c2r::skinny_transpose(
                data, m, n);
        } else {
			//fprintf(stdout, "r2c::skinny_transpose\n");
            inplace::detail::r2c::skinny_transpose(
                data, m, n);
        }
    } else if (small_m) {
        if (!row_major) {
			//fprintf(stdout, "r2c::skinny_transpose\n");
            inplace::detail::r2c::skinny_transpose(
                data, m, n);
        } else {
			//fprintf(stdout, "c2r::skinny_transpose\n");
            inplace::detail::c2r::skinny_transpose(
                data, m, n);
        }
    } else {
        bool m_greater = m > n;
        if (m_greater ^ row_major) {
			//fprintf(stdout, "r2c::transpose\n");
            inplace::r2c::transpose(row_major, data, m, n);
        } else {
			//fprintf(stdout, "c2r::transpose\n");
            inplace::c2r::transpose(row_major, data, m, n);
        }
    }*/
}

template void transpose(int*, int, int, int);
template void transpose(long long*, int, int, int);
template void transpose(float*, int, int, int);
template void transpose(double*, int, int, int);

}

#include "introspect.h"
#include "index.h"
#include "gcd.h"
#include "reduced_math.h"
#include "equations.h"
#include "smem.h"
#include <cassert>

#include <iostream>
#include <cstdio>
#include <cooperative_groups.h>

#include "save_array.h"
#include "cudacheck.h"

namespace inplace {
namespace detail {

namespace c2r {

struct fused_preop {
    reduced_divisor m;
    reduced_divisor b;
    __host__  fused_preop(int _m, int _b) : m(_m), b(_b) {}
    __host__ __device__
    int operator()(const int& i, const int& j) {
        return (int)m.mod(i + (int)b.div(j));
    }
};

//This shuffler exists for cases where m, n are large enough to cause overflow
struct long_shuffle {
    int m, n, k;
    reduced_divisor_64 b;
    reduced_divisor c;
    __host__
    long_shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
                                                   b(_n/_c), c(_c) {}
    int i;
    __host__ __device__ 
    void set_i(const int& _i) {
        i = _i;
    }
    __host__ __device__
    int f(const int& j) {
        int r = j + i * (n - 1);
        //The (int) casts here prevent unsigned promotion
        //and the subsequent underflow: c implicitly casts
        //int - unsigned int to
        //unsigned int - unsigned int
        //rather than to
        //int - int
        //Which leads to underflow if the result is negative.
        if (i - (int)c.mod(j) <= m - (int)c.get()) {
            return r;
        } else {
            return r + m;
        }
    }
    
    __host__ __device__
    int operator()(const int& j) {
        int fij = f(j);
        unsigned int fijdivc, fijmodc;
        c.divmod(fij, fijdivc, fijmodc);
        int term_1 = b.mod((long long)k * (long long)fijdivc);
        int term_2 = ((int)fijmodc) * (int)b.get();
        return term_1+term_2;
    }
};

struct fused_postop {
    reduced_divisor m;
    int n, c;
    __host__ 
    fused_postop(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
    __host__ __device__
    int operator()(const int& i, const int& j) {
        return (int)m.mod(i * n - (int)m.div(i * c) + j);
    }
};


}

namespace r2c {

struct fused_preop {
    reduced_divisor a;
    reduced_divisor c;
    reduced_divisor m;
    int q;
    __host__ 
    fused_preop(int _a, int _c, int _m, int _q) : a(_a) , c(_c), m(_m), q(_q) {}
    __host__ __device__ __forceinline__
    int p(const int& i) {
        int cm1 = (int)c.get() - 1;
        int term_1 = int(a.get()) * (int)c.mod(cm1 * i);
        int term_2 = int(a.mod(int(c.div(cm1+i))*q));
        return term_1 + term_2;
        
    }
    __host__ __device__
    int operator()(const int& i, const int& j) {
        int idx = m.mod(i + (int)m.get() - (int)m.mod(j));
        return p(idx);
    }
};

struct fused_postop {
    reduced_divisor m;
    reduced_divisor b;
    __host__  fused_postop(int _m, int _b) : m(_m), b(_b) {}
    __host__ __device__
    int operator()(const int& i, const int& j) {
        return (int)m.mod(i + (int)m.get() - (int)b.div(j));
    }
};


}

template<typename T, typename F, int U>
__global__ void long_row_shuffle(int d2, int d1, T* d, T* tmp, F s) {
	namespace cg = cooperative_groups;
	cg::grid_group g = cg::this_grid();
	row_major_index rm(d2, d1);
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	int grid_size = gridDim.x * blockDim.x;
	for (int i = 0; i < d2; i++) {
    	s.set_i(i);
		int j = global_id;
    	while(j + U * grid_size < d1) {
        	#pragma unroll
        	for(int k = 0; k < U; k++) {
            	tmp[j] = d[rm(i, s(j))];
            	j += grid_size;
        	}
    	}
    	while(j < d1) {
        	tmp[j] = d[rm(i, s(j))];
        	j += grid_size;
    	}

		g.sync();

		j = global_id;
		while(j + U * grid_size < d1) {
        	#pragma unroll
        	for(int k = 0; k < U; k++) {
				d[rm(i, j)] = tmp[j];
            	j += grid_size;
        	}
    	}
    	while(j < d1) {
			d[rm(i, j)] = tmp[j];
        	j += grid_size;
    	}
		g.sync();
	}
}

template<typename T, typename F>
__global__ void short_column_permute(int d2, int d1, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(d2, d1);
    row_major_index blk(blockDim.y, blockDim.x);
    int i = threadIdx.y; // One block tall by REQUIREMENT
    int grid_size = blockDim.x * gridDim.x;
    
    if (i < d2) {
        for(int j = threadIdx.x + blockIdx.x * blockDim.x;
            j < d1; j+= grid_size) {
            
            smem[blk(i, threadIdx.x)] = d[rm(i, j)];
            __syncthreads();
            d[rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
            __syncthreads();

        }   
    }
}

template<typename T, typename F>
void skinny_row_op(F s, int d2, int d1, T* d, T* tmp) {
	int n_threads = 256;
	int numBlocksPerSm;
	CudaSafeCall( cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, long_row_shuffle<T, F, 4>, n_threads, 0) );
	int n_blocks = numBlocksPerSm * n_sms();
	void *kernelArgs[] = {
		(void *)&d2,  (void *)&d1, (void *)&d, (void *)&tmp, (void *)&s
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)long_row_shuffle<T, F, 4>,
										  n_blocks, n_threads, kernelArgs) );
	
    /*for(int i = 0; i < d2; i++) {
        //long_row_shuffle<T, F, 4><<<(d1-1)/(256*4)+1,256>>>(d2, d1, i, d, tmp, s);
        //cudaMemcpy(d + d1 * i, tmp, sizeof(T) * d1, cudaMemcpyDeviceToDevice);

    }*/
}

template<typename T, typename F>
void skinny_col_op(F s, int d2, int d1, T* d) {
    int n_threads = 32;
    // XXX Potential optimization here: figure out how many blocks/sm
    // we should launch
    int n_blocks = n_sms()*8;
    dim3 grid_dim(n_blocks);
    dim3 block_dim(n_threads, d2);
    short_column_permute<<<grid_dim, block_dim,
        sizeof(T) * d2 * n_threads>>>(d2, d1, d, s);
}


namespace c2r {

template<typename T>
void skinny_transpose(T* data, int d1, int d2, int d3) {
    //std::cout << "Doing Skinny C2R transpose of " << d2 << ", " << d1 << std::endl;
	printf("Doing Skinny C2R transpose\n");

    assert(d2 <= 32);


    int c, t, k;
    extended_gcd(d2, d1, c, t);
    if (c > 1) {
        extended_gcd(d2/c, d1/c, t, k);
    } else {
        k = t;
    }
	
	T* tmp;
	CudaSafeCall( cudaMalloc(&tmp, sizeof(T) * d1) );

    for (int i = 0; i < d3; i++) {
		printf("i = %d\n", i);
		if (c > 1) {
        	skinny_col_op(fused_preop(d2, d1/c), d2, d1, data + i * d1 * d2);
    	}
    	skinny_row_op(long_shuffle(d2, d1, c, k), d2, d1, data + i * d1 * d2, tmp);
    	skinny_col_op(fused_postop(d2, d1, c), d2, d1, data + i * d1 * d2);
	}
	
	CudaSafeCall( cudaFree(tmp) );

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
	
	T* tmp;
	CudaSafeCall( cudaMalloc(&tmp, sizeof(T) * d1) );

	for (int i = 0; i < d3; i++) {
		skinny_col_op(fused_preop(d2/c, c, d2, q), d2, d1, data + i * d1 * d2);
		skinny_row_op(shuffle(d2, d1, c, 0), d2, d1, data + i * d1 * d2, tmp);
		if (c > 1) {
			skinny_col_op(fused_postop(d2, d1/c), d2, d1, data + i * d1 * d2);
		}
	}
	
	CudaSafeCall( cudaFree(tmp) );
}

template void skinny_transpose(int*, int, int, int);
template void skinny_transpose(long long*, int, int, int);
template void skinny_transpose(float*, int, int, int);
template void skinny_transpose(double*, int, int, int);

}



}
}

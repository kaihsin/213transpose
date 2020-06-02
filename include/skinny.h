#include "reduced_math.h"

namespace inplace {
namespace _2d {

namespace c2r {

template <typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int m, int n);

}

namespace r2c {

template <typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int m, int n);

}

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

}
}

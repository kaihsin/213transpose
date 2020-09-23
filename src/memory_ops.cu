#include <cooperative_groups.h>
#include "equations.h"
#include "index.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void memory_row_shuffle(int d3, int d2, int d1, T* d, T* tmp, F s) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
    row_major_index rm(d2, d1);
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    for (size_t k = 0; k < d3; k++) {
        size_t offset = k * d1d2;
        for (int i = 0; i < d2; i++) {
            s.set_i(i);
            int j = global_id;
            g.sync();
            while(j < d1) {
                tmp[j] = d[offset + rm(i, s(j))];
                j += grid_size;
            }

            j = global_id;
            g.sync();
            while(j < d1) {
                d[offset + rm(i, j)] = tmp[j];
                j += grid_size;
            }
        }
    }
}

template __global__ void memory_row_shuffle(int, int, int, float*, float*, c2r::shuffle);
template __global__ void memory_row_shuffle(int, int, int, double*, double*, c2r::shuffle);

template __global__ void memory_row_shuffle(int, int, int, int*, int*, c2r::shuffle);
template __global__ void memory_row_shuffle(int, int, int, long long*, long long*, c2r::shuffle);

template __global__ void memory_row_shuffle(int, int, int, float*, float*, r2c::shuffle);
template __global__ void memory_row_shuffle(int, int, int, double*, double*, r2c::shuffle);

template __global__ void memory_row_shuffle(int, int, int, int*, int*, r2c::shuffle);
template __global__ void memory_row_shuffle(int, int, int, long long*, long long*, r2c::shuffle);

}
}



namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void compress_row_shuffle(int d3, int d2, int d1, size_t smem_size, T* d, F s);

template<typename T, typename F>
__global__ void smem_row_shuffle(int d3, int d2, int d1, T* d, F s);

}
}

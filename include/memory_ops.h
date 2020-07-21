namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void memory_row_shuffle(int d3, int d2, int d1, T* d, T* tmp, F s);

}

}

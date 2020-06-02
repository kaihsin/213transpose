namespace inplace {
namespace _213 {

namespace c2r {

template <typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int d1, int d2, int d3);

}

namespace r2c {

template <typename T>
void skinny_transpose(cudaStream_t& stream, T* data, T* temp, int d1, int d2, int d3);

}

}
}
#include "equations.h"

namespace inplace {
namespace _2d {

template<typename T, typename F>
__global__ void memory_row_shuffle(int m, int n, T* d, T* tmp, F s);

}
}

#include <set>
#include <vector>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <thrust/transform.h>
#include <cuda.h>
#include "gcd.h"
#include "index.h"
#include "introspect.h"
#include "util.h"

namespace inplace {
namespace _2d {

template<typename T, typename F>
void scatter_permute(cudaStream_t& stream, F f, int m, int n, T* data, T* tmp);

}
}

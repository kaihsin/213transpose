#pragma once
#include <stdio.h>
#include <iostream>
#include <iterator>
#include <cuda.h>
#include "index.h"
#include "util.h"

namespace inplace {
namespace _2d {

template<typename F, typename T>
void rotate(cudaStream_t& stream, F f, int m, int n, T* data);

}
}

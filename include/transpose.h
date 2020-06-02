#pragma once

#include "introspect.h"
#include "index.h"
#include "equations.h"

namespace inplace {

namespace _2d {

namespace c2r {
void transpose(cudaStream_t& stream, bool row_major, float* data, float* temp, int m, int n);
void transpose(cudaStream_t& stream, bool row_major, double* data,  double* temp, int m, int n);
}
namespace r2c {
void transpose(cudaStream_t& stream, bool row_major, float* data, float* temp, int m, int n);
void transpose(cudaStream_t& stream, bool row_major, double* data, double* temp, int m, int n);
}

void transpose(cudaStream_t& stream, bool row_major, float* data, float* temp, int m, int n);
void transpose(cudaStream_t& stream, bool row_major, double* data, double* temp, int m, int n);

}

}


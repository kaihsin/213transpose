#pragma once

#define CudaSafeCall(err) __cudaSafeCall( err, __FILE__, __LINE__ )
void __cudaSafeCall(cudaError err, const char *file, const int line);

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include "tensor_util.h"

template<typename T>
TensorUtil<T>::TensorUtil(FILE* _fp, int _rank, int* _dim, int* _permutation): fp(_fp), rank(_rank) {
	size_t dim_byte_size = (size_t)rank * sizeof(int);
	dim = (size_t*)malloc(dim_byte_size);
	permutation = (size_t*)malloc(dim_byte_size);
	stride = (size_t*)malloc(dim_byte_size);
	for (int i = 0; i < rank; i++) {
		dim[i] = (size_t)_dim[i];
		permutation[i] = (size_t)_permutation[i];
	}
	stride[0] = 1;
	std::partial_sum(dim, dim + rank - 1, stride + 1, std::multiplies<size_t>());
	vol = 1;
	for (int i = 0; i < rank; i++) {
		vol *= dim[i];
	}
}

template TensorUtil<int>::TensorUtil(FILE*, int, int*, int*);
template TensorUtil<long long>::TensorUtil(FILE*, int, int*, int*);
template TensorUtil<float>::TensorUtil(FILE*, int, int*, int*);
template TensorUtil<double>::TensorUtil(FILE*, int, int*, int*);

template<typename T>
TensorUtil<T>::~TensorUtil() {
	free(dim);
	free(permutation);
	free(stride);
}

template TensorUtil<int>::~TensorUtil();
template TensorUtil<long long>::~TensorUtil();
template TensorUtil<float>::~TensorUtil();
template TensorUtil<double>::~TensorUtil();

template<typename T>
void TensorUtil<T>::init_data(T* data) {
	for (size_t i = 0; i < vol; i++) {
		data[i] = i;
	}
}

template void TensorUtil<int>::init_data(int*);
template void TensorUtil<long long>::init_data(long long*);
template void TensorUtil<float>::init_data(float*);
template void TensorUtil<double>::init_data(double*);

template<typename T>
void TensorUtil<T>::print_mat(T* data) {
	for (size_t i = 0; i < vol; i++) {
		fprintf(fp, "%-10d", (int)data[i]);
		for (int j = 1; j < rank; j++) {
			if ((i + 1) % stride[j] == 0) fprintf(fp, "\n");
		}
	}
}

template void TensorUtil<int>::print_mat(int*);
template void TensorUtil<long long>::print_mat(long long*);
template void TensorUtil<float>::print_mat(float*);
template void TensorUtil<double>::print_mat(double*);

template<typename T>
void TensorUtil<T>::write_file(T* data) {
	if (fp == stdout) {
		print_mat(data);
	}
	else {
		fwrite(data, sizeof(T), vol, fp);
	}
}

template void TensorUtil<int>::write_file(int*);
template void TensorUtil<long long>::write_file(long long*);
template void TensorUtil<float>::write_file(float*);
template void TensorUtil<double>::write_file(double*);

template<typename T>
void TensorUtil<T>::seq_tt(T* ans, T* data) {
	for (size_t idx_s = 0; idx_s < vol; idx_s++) {
		size_t idx_d = 0;
		for (int i = 0; i < rank; i++) {
			size_t stride_d = 1;
			for (int j = 0; permutation[j] != i; j++) {
				stride_d *= dim[permutation[j]];
			}
			idx_d += ((idx_s / stride[i]) % dim[i]) * stride_d;
		}
		ans[idx_d] = data[idx_s];
	}
}

template void TensorUtil<int>::seq_tt(int*, int*);
template void TensorUtil<long long>::seq_tt(long long*, long long*);
template void TensorUtil<float>::seq_tt(float*, float*);
template void TensorUtil<double>::seq_tt(double*, double*);

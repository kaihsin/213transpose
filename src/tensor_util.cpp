#include <cstdio>
#include "tensor_util.h"

namespace _2d {

void init_data(int* dim, int* h_data) {
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			int idx = j + i * dim[1];
			h_data[idx] = idx;
		}
	}
}

void print_mat(FILE* fp, int* dim, int* h_data) {
	for (int i = 0; i < dim[0]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			fprintf(fp, "%-10d", h_data[j + i * dim[1]]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
}
}

namespace _213 {

void init_data(int* dim, int* h_data) {
	for (int i = 0; i < dim[2]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[0]; k++) {
				long long idx = (long long)k + (long long)j * dim[0] + (long long)i * dim[0] * dim[1];
				h_data[idx] = idx;
			}
		}
	}
}

template<typename T>
void init_data(T* h_data, size_t vol) {
	for (size_t i = 0; i < vol; i++) {
		h_data[i] = i;
	}
}

template void init_data(int*, size_t);
template void init_data(long long*, size_t);
template void init_data(float*, size_t);
template void init_data(double*, size_t);

void print_mat(FILE* fp, int* dim, int* h_data) {
	for (int i = 0; i < dim[2]; i++) {
		for (int j = 0; j < dim[1]; j++) {
			for (int k = 0; k < dim[0]; k++) {
				long long idx = (long long)k + (long long)j * dim[0] + (long long)i * dim[0] * dim[1];
				fprintf(fp, "%-10d", h_data[idx]);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
}

template<typename T>
void print_mat(FILE* fp, T* h_data, int* dim, size_t vol) {
	for (size_t i = 0; i < vol; i++) {
		fprintf(fp, "%-10d", h_data[i]);
		if ((i + 1) % dim[0] == 0) fprintf(fp, "\n");
		if ((i + 1) % (dim[0] * dim[1]) == 0) fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
}

template void print_mat(FILE*, int*, int*, size_t);
template void print_mat(FILE*, long long*, int*, size_t);
template void print_mat(FILE*, float*, int*, size_t);
template void print_mat(FILE*, double*, int*, size_t);

void seq_tt(int*& ans, int*& data, int d1, int d2, int d3) {
	for (int k = 0; k < d3; k++) {
		for (int j = 0; j < d1; j++) {
			for (int i = 0; i < d2; i++) {
				long long idx_s = (long long)j + (long long)i * d1 + (long long)k * d1 * d2;
				long long idx_d = (long long)i + (long long)j * d2 + (long long)k * d2 * d1;
				ans[idx_d] = data[idx_s];
			}
		}
	}
}

template<typename T>
void seq_tt(T*& ans, T*& data, int* dim, size_t vol) {
	for (size_t idx_s = 0; idx_s < vol; idx_s++) {
		size_t j = idx_s % dim[0];
		size_t i = (idx_s / dim[0]) % dim[1];
		size_t k = idx_s / (dim[0] * dim[1]);
		size_t idx_d = i + j * dim[1] + k * dim[1] * dim[0];
		ans[idx_d] = data[idx_s];
	}
}

template void seq_tt(int*&, int*&, int*, size_t);
template void seq_tt(long long*&, long long*&, int*, size_t);
template void seq_tt(float*&, float*&, int*, size_t);
template void seq_tt(double*&, double*&, int*, size_t);
}






template<typename T>
void _213TensorUtil<T>::init_data(T* data) {
	for (size_t i = 0; i < this->vol; i++) {
		data[i] = i;
	}
}

template void _213TensorUtil<int>::init_data(int*);
template void _213TensorUtil<long long>::init_data(long long*);
template void _213TensorUtil<float>::init_data(float*);
template void _213TensorUtil<double>::init_data(double*);

template<typename T>
void _213TensorUtil<T>::print_mat(T* data) {
	for (size_t i = 0; i < this->vol; i++) {
		fprintf(this->fp, "%-10d", data[i]);
		if ((i + 1) % this->dim[0] == 0) fprintf(this->fp, "\n");
		if ((i + 1) % (this->dim[0] * this->dim[1]) == 0) fprintf(this->fp, "\n");
	}
	fprintf(this->fp, "\n");
}

template void _213TensorUtil<int>::print_mat(int*);
template void _213TensorUtil<long long>::print_mat(long long*);
template void _213TensorUtil<float>::print_mat(float*);
template void _213TensorUtil<double>::print_mat(double*);

template<typename T>
void _213TensorUtil<T>::seq_tt(T* ans, T* data) {
	for (size_t idx_s = 0; idx_s < this->vol; idx_s++) {
		size_t j = idx_s % this->dim[0];
		size_t i = (idx_s / this->dim[0]) % this->dim[1];
		size_t k = idx_s / (this->dim[0] * this->dim[1]);
		size_t idx_d = i + j * this->dim[1] + k * this->dim[1] * this->dim[0];
		ans[idx_d] = data[idx_s];
	}
}

template void _213TensorUtil<int>::seq_tt(int*, int*);
template void _213TensorUtil<long long>::seq_tt(long long*, long long*);
template void _213TensorUtil<float>::seq_tt(float*, float*);
template void _213TensorUtil<double>::seq_tt(double*, double*);
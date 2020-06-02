#include <cstdio>

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
}
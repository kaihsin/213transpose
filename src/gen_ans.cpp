#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "tensor_util.h"

template<typename T>
void gen(_213TensorUtil<T>& _213tu) {
	size_t vol = _213tu.vol;
	T* i_data = (T*)malloc(vol * sizeof(T));
	_213tu.init_data(i_data);
	
	T* o_data = (T*)malloc(vol * sizeof(T));
	_213tu.seq_tt(o_data, i_data);
	_213tu.print_mat(o_data);
	
	free(i_data);
	free(o_data);
}

int main(int argc, char** argv) {
	int dim[3];
	dim[0] = atoi(argv[1]); // d1
	dim[1] = atoi(argv[2]); // d2
	dim[2] = atoi(argv[3]); // d3
	int type_size = atoi(argv[4]);
	//printf("Size of data type = %d bytes\n", type_size);
	FILE* fp = (argc == 6)? fopen(argv[5], "wb") : stdout;
	size_t vol = (size_t)dim[0] * dim[1] * dim[2];
	if (type_size == 4) {
		_213TensorUtil<int> _213tu(fp, dim, vol, sizeof(dim));
		gen<int>(_213tu);
	}
	else {
		_213TensorUtil<long long> _213tu(fp, dim, vol, sizeof(dim));
		gen<long long>(_213tu);
	}
	
	return 0;
}
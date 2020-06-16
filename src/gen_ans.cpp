#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include "tensor_util.h"

template<typename T>
void gen(TensorUtil<T>& tu) {
	size_t vol = tu.vol;
	T* i_data = (T*)malloc(vol * sizeof(T));
	tu.init_data(i_data);
	
	T* o_data = (T*)malloc(vol * sizeof(T));
	tu.seq_tt(o_data, i_data);
	tu.write_file(o_data);
	
	free(i_data);
	free(o_data);
}

int main(int argc, char** argv) {
	assert(argc == 5 || argc == 6);
	int dim[3];
	dim[0] = atoi(argv[1]); // d1
	dim[1] = atoi(argv[2]); // d2
	dim[2] = atoi(argv[3]); // d3
	int type_size = atoi(argv[4]);
	
	FILE* fp = (argc == 6)? fopen(argv[5], "wb") : stdout;
	int permutation[3] = {1, 0, 2};
	if (type_size == 4) {
		TensorUtil<int> tu(fp, 3, dim, permutation);
		gen<int>(tu);
	}
	else {
		TensorUtil<long long> tu(fp, 3, dim, permutation);
		gen<long long>(tu);
	}
	
	if (fp != stdout) fclose(fp);
	
	return 0;
}
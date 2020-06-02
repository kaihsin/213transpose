#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "transpose_213.h"
#include "tensor_util.h"

int main(int argc, char** argv) {
	printf("213 inplace tensor reshape\n");
	
	int h_dim[3];
	int d1 = h_dim[0] = atoi(argv[1]);
	int d2 = h_dim[1] = atoi(argv[2]);
	int d3 = h_dim[2] = atoi(argv[3]);
	
	size_t vol = (long long)d1 * d2 * d3;
	printf("Data Size = %lld Bytes\n", (long long)vol * sizeof(int));
	int* h_data = (int*)malloc(vol * sizeof(int));
	_213::init_data(h_dim, h_data);
	int* ans = (int*)malloc(vol * sizeof(int));
	memcpy(ans, h_data, sizeof(int) * vol);
	_213::seq_tt(ans, h_data, d1, d2, d3);
	//print_mat(stdout, h_dim, h_data);
	
	inplace::_213::transpose(h_data, d1, d2, d3);

	//print_mat(stdout, h_dim, h_data);
	//print_mat(stdout, h_dim, ans);
	if (!memcmp(ans, h_data, vol * sizeof(int))) printf("Correct\n");
	else printf("Error\n");
	
	free(h_data);
	
	return 0;
}
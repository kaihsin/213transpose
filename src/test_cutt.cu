#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda.h>
#include "cutt.h"
#include "tensor_util.h"
#include "cudacheck.h"

#define cuttCheck(stmt) do {                                 \
  cuttResult err = stmt;                            \
  if (err != CUTT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

template<typename T>
float cutt_plan(T* idata, T* odata, int* dim, int* permutation, size_t dataSize) {
	cudaEvent_t start, stop;
	CudaSafeCall( cudaEventCreate(&start) );
	CudaSafeCall( cudaEventCreate(&stop) );
	CudaSafeCall( cudaEventRecord(start, 0) );
	
	cuttHandle plan;
	cuttCheck(cuttPlan(&plan, 3, dim, permutation, sizeof(T), 0));
    
    int dev;
    CudaSafeCall( cudaGetDevice(&dev) );
    CudaSafeCall( cudaMemPrefetchAsync(idata, dataSize, dev, 0) );
	cuttCheck(cuttExecute(plan, idata, odata));
	
	CudaSafeCall( cudaDeviceSynchronize() );
	CudaSafeCall( cudaEventRecord(stop, 0) );
	CudaSafeCall( cudaEventSynchronize(stop) );
	float t;
	CudaSafeCall( cudaEventElapsedTime(&t, start, stop) );

	cuttCheck(cuttDestroy(plan));
	return t;
}

template<typename T>
float cutt_plan_measure(T* idata, T* odata, int* dim, int* permutation, size_t dataSize) {
	cudaEvent_t start, stop;
	CudaSafeCall( cudaEventCreate(&start) );
	CudaSafeCall( cudaEventCreate(&stop) );
	CudaSafeCall( cudaEventRecord(start, 0) );
	
	cuttHandle plan;
	cuttPlanMeasure(&plan, 3, dim, permutation, sizeof(T), 0, idata, odata);
    int dev;
    CudaSafeCall( cudaGetDevice(&dev) );
    CudaSafeCall( cudaMemPrefetchAsync(idata, dataSize, dev, 0) );
	cuttCheck(cuttExecute(plan, idata, odata));
	
	CudaSafeCall( cudaDeviceSynchronize() );
	CudaSafeCall( cudaEventRecord(stop, 0) );
	CudaSafeCall( cudaEventSynchronize(stop) );
	float t;
	CudaSafeCall( cudaEventElapsedTime(&t, start, stop) );

	cuttCheck(cuttDestroy(plan));
	return t;
}

template<typename T>
void test_cutt(TensorUtil<T>& tu) {
	size_t& vol = tu.vol;
	T* i_data = NULL;
	size_t dataSize = vol * sizeof(T);
	CudaSafeCall( cudaMallocManaged(&i_data, dataSize) );
	tu.init_data(i_data);
	
	int dim[3] = {(int)tu.dim[0], (int)tu.dim[1], (int)tu.dim[2]};
	int permutation[3] = {1, 0, 2};
	
	//T* ans = (T*)malloc(dataSize);
	//tu.seq_tt(ans, i_data);

	T* o_data = NULL;
	CudaSafeCall( cudaMallocManaged(&o_data, dataSize) );
	float t1 = cutt_plan(i_data, o_data, dim, permutation, dataSize);
	
	CudaSafeCall( cudaFree(i_data) );
	CudaSafeCall( cudaFree(o_data) );
	
	CudaSafeCall( cudaMallocManaged(&i_data, dataSize) );
	tu.init_data(i_data);
	CudaSafeCall( cudaMallocManaged(&o_data, dataSize) );
	
	float t2 = cutt_plan_measure(i_data, o_data, dim, permutation, dataSize);

	//printf("t1 = %.5f\nt2 = %.5f\nTime: %.5fms\n", t1, t2, std::min(t1, t2));
    printf("%.5f\n", std::min(t1, t2));
	
	CudaSafeCall( cudaFree(i_data) );
	CudaSafeCall( cudaFree(o_data) );
	//free(ans);
}

int main(int argc, char** argv) {
	int dim[3];
	dim[0] = atoi(argv[1]);
	dim[1] = atoi(argv[2]);
	dim[2] = atoi(argv[3]);
	int permutation[3] = {1, 0, 2};
	size_t dataSize = (long long)dim[0] * dim[1] * dim[2] * sizeof(int);
	//printf("Data Size = %lld bytes\n", (long long)dataSize);
	
	int type_size = atoi(argv[4]);
	FILE* fp = (argc == 6)? fopen(argv[5], "wb") : stdout;
	if (type_size == 4) {
		TensorUtil<int> tu(fp, 3, dim, permutation);
		test_cutt<int>(tu);
	}
	else {
		TensorUtil<long long> tu(fp, 3, dim, permutation);
		test_cutt<long long>(tu);
	}
	if (fp != stdout) fclose(fp);
}

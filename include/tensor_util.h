#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace _2d {
void init_data(int* dim, int* h_data);
void print_mat(FILE* fp, int* dim, int* h_data);
}

namespace _213 {
void init_data(int* dim, int* h_data);
void print_mat(FILE* fp, int* dim, int* h_data);
void seq_tt(int*& ans, int*& data, int d1, int d2, int d3);

using size_t = std::size_t;
template<typename T> void init_data(T* h_data, size_t vol);
template<typename T> void print_mat(FILE* fp, T* h_data, int* dim, size_t vol);
template<typename T> void seq_tt(T*& ans, T*& data, int* dim, size_t vol);
}

template <typename T>
class TensorUtil {
public:
	FILE* fp;
	int* dim;
	size_t rank;
	size_t vol;
	TensorUtil(FILE* _fp, int* _dim, size_t _vol, size_t _rank): fp(_fp), vol(_vol), rank(_rank) {
		this->dim = (int*)malloc(rank * sizeof(int));
		memcpy(this->dim, _dim, rank * sizeof(int));
	}
	
	~TensorUtil() {
		free(this->dim);
	}
};

template <typename T>
class _213TensorUtil: public TensorUtil<T> {
public:
	_213TensorUtil(FILE* _fp, int* _dim, size_t _vol, size_t _rank) : TensorUtil<T>(_fp, _dim, _vol, _rank) {}
	void init_data(T* data);
	void print_mat(T* data);
	void seq_tt(T* ans, T* data);
};
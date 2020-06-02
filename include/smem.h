namespace inplace {
namespace _2d {

template<class T>
struct shared_memory{
    __device__ inline operator T*() const {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

}
}

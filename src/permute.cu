#include <vector>
#include "gcd.h"
#include "index.h"
#include "introspect.h"
#include "util.h"
#include "equations.h"
#include <utility>
#include <algorithm>
#include <cstdio>

namespace inplace {
namespace detail {

using vector_pair = std::vector<std::pair<int, int> >;
    
template<typename Fn>
void scatter_cycles(Fn f, int* heads, int* lens, vector_pair& vp) {
    std::vector<bool> visited(f.len(), false);
    for (size_t head = 0; head < visited.size(); head++) {
        if (visited[head]) continue;
        visited[head] = true;
        int idx = head;
        int dest = f(idx);
        if (idx != dest) {
            int start = idx;
            int len = 1;
            while (dest != start) {
                idx = dest;
                visited[idx] = true;
                dest = f(idx);
                len++;
            }
            vp.push_back(std::make_pair(len, head));
        }
    }
    /*thrust::counting_iterator<int> i(0);
    std::set<int> unvisited(i, i+f.len());
    while(!unvisited.empty()) {
        int idx = *unvisited.begin();
        //printf("idx = %d\n", idx);
        unvisited.erase(unvisited.begin());
        int dest = f(idx);
        if (idx != dest) {
            //heads.push_back(idx);
            int head = idx;
            int start = idx;
            int len = 1;
            //std::cout << "Cycle: " << start << " " << dest << " ";
            while(dest != start) {
                idx = dest;
                unvisited.erase(idx);
                dest = f(idx);
                //printf("f(%d) = %d\n", idx, dest);
                len++;
                //std::cout << dest << " ";
            }
            //std::cout << std::endl;
            //lens.push_back(len);
            vp.push_back(std::make_pair(len, head));
        }
    }*/
    
    std::make_heap(vp.begin(), vp.end());
   // sort(vp.begin(), vp.end());
    //reverse(vp.begin(), vp.end());
    
    for (size_t i = 0; i < vp.size(); i++) {
        heads[i] = vp[i].second;
        lens[i] = vp[i].first;
    }
}

template<typename T, typename F, int U>
__global__ void cycle_row_permute(F f, T* data, int* heads,
                                  int* lens, int n_heads, int d3) {
    int d1 = f.n;
    int d2 = f.m;
    row_major_index rm(d2, d1);
    size_t d1d2 = (size_t)d1 * (size_t)d2;
    //size_t nhd3 = (size_t)n_heads * (size_t)d3;
    
    //for (size_t hk = blockIdx.x; hk < nhd3; hk += gridDim.x) {
        size_t hk = blockIdx.x;
        size_t k = hk / (size_t)n_heads;
        size_t offset = k * d1d2;
        size_t h = hk % n_heads;
        int i = heads[h];
        int l = lens[h];
        for (int j = threadIdx.x; j < d1; j += blockDim.x) {
            //unroll_cycle_row_permute<T, F, U>(f, rm, data + offset, i, j, l);
            T src = data[offset + rm(i, j)];
            T loaded[U+1];
            loaded[0] = src;
            for(int k = 0; k < l / U; k++) {
                int rows[U];
        #pragma unroll
                for(int x = 0; x < U; x++) {
                    i = f(i);
                    rows[x] = i;
                }
        #pragma unroll
                for(int x = 0; x < U; x++) {
                    loaded[x+1] = data[offset + rm(rows[x], j)];
                }
        #pragma unroll
                for(int x = 0; x < U; x++) {
                    data[offset + rm(rows[x], j)] = loaded[x];
                }
                loaded[0] = loaded[U];
            }
            T tmp = loaded[0];
            for(int k = 0; k < l % U; k++) {
                i = f(i);
                T new_tmp = data[offset + rm(i, j)];
                data[offset + rm(i, j)] = tmp;
                tmp = new_tmp;
            }
        }
    //}
    /*int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int d1 = f.n;
    int d2 = f.m;
    row_major_index rm(d2, d1);
    if ((j < d1) && (h < n_heads)) {
        size_t d1d2 = (size_t)d1 * (size_t)d2;
        for (size_t k = 0; k < d3; k++) {
            size_t offset = k * d1d2;
            int i = heads[h];
            int l = lens[h];
            unroll_cycle_row_permute<T, F, U>(f, rm, data + offset, i, j, l);
        }
    }*/
}

template<typename T, typename F>
void scatter_permute(F f, int d3, int d2, int d1, T* data, int* tmp) {
    int* d_heads = tmp;
    int* d_lens = tmp + d2 / 2;
    
    vector_pair vp;
    scatter_cycles(f, d_heads, d_lens, vp);
    
    /*cudaMemcpy(d_heads, heads.data(), sizeof(int)*heads.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lens, lens.data(), sizeof(int)*lens.size(),
               cudaMemcpyHostToDevice);*/

    int n_threads = 64;
    
    int n_heads = (int)vp.size();
    int active_blocks = get_num_block(cycle_row_permute<T, F, 4>, n_threads, 0);
    int n_blocks = n_heads * d3; //min(n_heads * d3, active_blocks);
    //printf("active_blocks = %d\n", active_blocks);
    
    cycle_row_permute<T, F, 4><<<n_blocks, n_threads>>>(f, data, d_heads, d_lens, n_heads, d3);
    
    /*int n_threads_x = 256;
    int n_threads_y = 1024/n_threads_x;
    
    int n_blocks_x = div_up(d1, n_threads_x);
    int n_blocks_y = div_up(heads.size(), n_threads_y);
    cycle_row_permute<T, F, 4>
    <<<dim3(n_blocks_x, n_blocks_y),
    dim3(n_threads_x, n_threads_y)>>>
    (f, data, d_heads, d_lens, heads.size(), d3);*/
}


template void scatter_permute(c2r::scatter_postpermuter, int, int, int, float*, int*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, int, double*, int*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, int, int*, int*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, int, long long*, int*);

template void scatter_permute(r2c::scatter_prepermuter, int, int, int, float*, int*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, int, double*, int*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, int, int*, int*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, int, long long*, int*);


}
}

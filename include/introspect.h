#pragma once
namespace inplace {

struct introspect {
    int device;
    cudaDeviceProp properties;
    introspect() {
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&properties, device);
    }
};

int n_sms();
int n_ctas();
int n_threads();
size_t gpu_memory_size();
size_t shared_mem_per_block();
int current_sm();

}

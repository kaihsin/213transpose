#include "util.h"

int get_num_thread(int d1) {
    int msb = static_cast<int>(log2(d1)); // most significant bit
    unsigned n_threads = static_cast<unsigned>(2 << msb);
    unsigned lim = 1024;
    return static_cast<int>(min(n_threads, lim));
}

void Timer::start() {
    start_tp = clock::now();
}
void Timer::stop() {
    stop_tp = clock::now();
}
double Timer::elapsed_time() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_tp - start_tp).count();
}
int get_num_thread(int d1) {
    int msb = static_cast<int>(log2(d1)); // most significant bit
    unsigned n_threads = static_cast<unsigned>(2 << msb);
    unsigned lim = 1024;
    return static_cast<int>(min(n_threads, lim));
}

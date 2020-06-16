#pragma once

namespace inplace {

namespace c2r {

template <typename T> void transpose(T* data, int d1, int d2, int d3);

}

namespace r2c {

template <typename T>void transpose(T* data, int d1, int d2, int d3);

}

template <typename T>void transpose(T* data, int d1, int d2, int d3);

}


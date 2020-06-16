#pragma once
namespace inplace {
namespace detail {

namespace c2r {

template <typename T>
void skinny_transpose(T* data, int d1, int d2, int d3);

}

namespace r2c {

template <typename T>
void skinny_transpose(T* data, int d1, int d2, int d3);

}

}
}

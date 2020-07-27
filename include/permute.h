#pragma once

namespace inplace {
namespace detail {

template<typename T, typename F>
void scatter_permute(F f, int d3, int d2, int d1, T* data);

}
}

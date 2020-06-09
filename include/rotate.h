#pragma once
#include <stdio.h>
#include <iostream>
#include <iterator>
#include "index.h"
#include "util.h"

namespace inplace {
namespace detail {

template<typename F, typename T>
void rotate(F f, int m, int n, T* data);

}
}

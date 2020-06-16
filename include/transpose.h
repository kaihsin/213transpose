#pragma once

namespace inplace {

<typename T>
void transpose(T* data, int d1, int d2, int d3);

namespace c2r {

<typename T>
void transpose(T* data, int d1, int d2, int d3);

}

namespace r2c {

<typename T>
void transpose(T* data, int d1, int d2, int d3);

}

}


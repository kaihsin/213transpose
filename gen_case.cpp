#include <cstdio>
#include <cassert>
#include <vector>
#include <string>

using namespace std;

const size_t data_size = 1375000000; // 11 GB / 2 / 4
char str[1024];

int main(int argc, char** argv) {
    bool dump_raw = false;
    if (argc == 2) dump_raw = true;
	vector<string> vec;
    printf("PARAMS=(");
	size_t d1 = 34, d3 = 2, d2 = data_size / d1 / d3;
    int type_size = 4;

	while (d1 < d2) {
        size_t vol = d1 * d2 * d3;
        assert(vol <= data_size);
        printf("\"%zu %zu %zu %d\"", d1, d2, d3, type_size);
		sprintf(str, "(%zu, %zu, %zu)\n", d1, d2, d3);
        vec.push_back(str);
		d2 >>= 1;
		d1 <<= 1;
        if (d1 < d2) printf("\\\n\t");
        else printf("\n)\n");
	}
    
    if (dump_raw) {
        for (auto s: vec) {
            printf("%s", s.c_str());
        }
    }
    
    return 0;
}

#include <iostream>

using namespace std;

int main() {
	cout << "PARAMS=(";
	int d1 = 343750000, d2 = 2, d3 = 2;

	while (d1 > 1) {
		cout << "\t\"" << d1 << " 2 " << d3 << " 4" << "\"\\\n";
		d1 >>= 1;
		d3 <<= 1;
	}
	cout << ")\n";
}

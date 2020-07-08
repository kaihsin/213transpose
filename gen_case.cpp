#include <iostream>

using namespace std;

int main() {
	cout << "PARAMS=(";
	int d2 = 178125000, d3 = 2;

	while (d2 >= 1) {
		cout << "\"2 " << d2 << " " << d3 << " 4" << "\"\\\n";
		d2 >>= 1;
		d3 <<= 1;
	}
	cout << ")\n";
}

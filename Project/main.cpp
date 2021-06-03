#include <iostream>
#include "matrix.hpp"

using namespace std;

int main() {
    Matrix<int> m(3, 3), m1(2, 2);
    int a[] = { 1,2,3,4,5,6,7,8,9 };
    int b[] = { 1,2,3,4 };
    m.set(9, a);
    m1.set(4, b);
    Matrix<int> c = m.Convolve(m1);
    //snow 2
    c.print();
    return 0;
}

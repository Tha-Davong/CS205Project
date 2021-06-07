#include <iostream>
#include "matrix.hpp"
#include "sparseMatrix.hpp"
#include "templateUtil.h"

using namespace std;

int main() {
    std::complex<double> b[25];
    for(int i = 0; i < 25; ++i) {
        b[i].real(i+1.5);
        b[i].imag(25-i+1.5);
    }
    Matrix<std::complex<double>> c(4,4);
    c.set(16,b);
    c.print();

    int a[25];
    for (int i = 0; i < 25; ++i) {
        a[i] = i + 1;
    }
    Matrix<int> m(4, 4);
    sparseMatrix<int> k(4,4);

    m.set(16, a);
    sparseMatrix<int> k1(m);
    k1.print();
    //m.print();
    m.print();
    Matrix<int> m1 = m.Slice(0, 3, 2, 0, 3, 2);
    m1.print();
    cout << "here" << endl;
    return 0;
}
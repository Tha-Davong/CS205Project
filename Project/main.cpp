#include <iostream>
#include "matrix.hpp"
#include "sparseMatrix.hpp"
#include "templateUtil.h"
#include <complex>

using namespace std;

int main() {

    //this is the testing branch
    Matrix<int> m1(4, 4), m2(2, 3);
    Matrix<complex<int>> m3(2, 2);
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[] = { 1,2,3,4,5,6};
    complex<int> c[] = {complex<int>(1,1), complex<int>(2,2), complex<int>(3,3), complex<int>(4,4)};
    m1.set(16, a);
    m1.print();
    m1.Max(0).print();
    
    
    
    
    return 0;
}

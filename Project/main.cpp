#include <iostream>
#include "matrix.hpp"
#include "sparseMatrix.hpp"
#include "templateUtil.h"
#include <complex>

using namespace std;

int main() {

    //this is the testing branch
    Matrix<int> m1(4, 4), m2(2, 3);
    Matrix<complex<double>> m3(4, 3);
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[] = { 1,2,3,4,5,6};
    complex < double > c[] = { complex<double>(1,1), complex<double>(2,2), complex<double>(3,3), complex<double>(4,4),
                        complex<double>(5,5), complex<double>(6,6), complex<double>(7,7), complex<double>(8,8), 
                            complex<double>(9,9), complex<double>(10,10), complex<double>(11,11), complex<double>(12,12) };
    m1.set(16, a);
    //m1.print();
    m3.set(12, c);
    m1.print();
    m1.Min(0).print();
   
    
    

    
    
    
    
    
    
    return 0;
}

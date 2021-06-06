#include <iostream>
#include "matrix.hpp"

using namespace std;

int main() {
    //this is the testing branch
    Matrix<int> m1(4, 4), m2(2, 3);
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[] = { 1,2,3,4,5,6};
    m1.set(16, a);
    m2.set(6, b);
    Matrix<int> c = m1.Convolve(m2);
    
    c.print();
    cout << endl;
    c.reshape(2,3);
    c.print();
    c.Avg(1).print();
    return 0;
}

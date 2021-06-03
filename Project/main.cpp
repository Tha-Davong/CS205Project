#include <iostream>
#include "matrix.hpp"

using namespace std;

int main() {
    //this is the testing branch
    Matrix<int> m1(4, 4), m2(2, 1);
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[] = { 1,2};
    m1.set(16, a);
    m2.set(2, b);
    Matrix<int> c = m1.Convolve(m2);
    
    c.print();
    cout << endl;
    c.reshape(4,3);
    c.print();
   cout << endl;
  Matrix<int> slice = c.Slice(0,1,1,1,2,1);
  slice.print();
    return 0;
}

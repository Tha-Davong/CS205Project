#include <iostream>
#include "matrix.hpp"

using namespace std;

int main() {
    //this is the testing branch
    Matrix<int> m1(4, 4), m2(2, 3);
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[] = { 1,2,3,4,5,6};
   
    
    try {
        //m2.Convolve(m1);
        m1.set(16, a);
        m2.set(6, b);
        m1.print();
        cout << endl;
        m1.Slice(0, 2, 1, 2, 0, 1).print();
        
    }
    catch(exception e){
        cout << e.what();
    }
    
    
    return 0;
}

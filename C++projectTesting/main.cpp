#include "matrix.hpp"
#include <iostream>

using namespace std;

template<typename T>
void printMatrix(const matrix::Matrix<T> &mat) {
    for (int i = 0; i < mat.getRows(); ++i) {
        for (int j = 0; j < mat.getCols(); ++j) {
            cout << mat.get(i, j) << " ";
        }
        cout << endl;
    }
}

int main() {
    matrix::Matrix<int> m(4,4);
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    m.set(16,a);
    matrix::Matrix<int> m1 = m.slice(-90,3,1,0,3,1);


    printMatrix(m1);
     
    return 0;
}


#include <iostream>
#include "matrix.hpp"

using namespace std;

int main() {
    int a[25];
    for (int i = 0; i < 25; ++i) {
        a[i] = i + 1;
    }
    Matrix<int> m(4, 4);
    m.set(16, a);
    //m.print();
    m.print();
    Matrix<int> m1 = m.Slice(0, 3, 2, 0, 3, 2);
    m1.print();
    cout << "here" << endl;
    return 0;
}
//
// Created by Vithlaithla Long on 4/6/21.
//
#include "matrix.hpp"

#ifndef CS205PROJECT_SPARSEMATRIX_HPP
#define CS205PROJECT_SPARSEMATRIX_HPP
template<typename T>
class sparseMatrix {
private:
    int rows, cols;
    std::vector<int> row_index;
    std::vector<int> col_index;
    std::vector<T> elements;
    int size;
public:
    sparseMatrix(int row, int col);
    sparseMatrix(Matrix<T> & matrix);
    sparseMatrix(const sparseMatrix<T> &);
    ~sparseMatrix();

    T get(int row, int col);
    sparseMatrix & set(T val, int row, int col);
};



#endif //CS205PROJECT_SPARSEMATRIX_HPP

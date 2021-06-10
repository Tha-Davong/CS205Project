//
// Created by Vithlaithla Long on 4/6/21.
//

#ifndef CS205PROJECT_SPARSEMATRIX_HPP
#define CS205PROJECT_SPARSEMATRIX_HPP
#include "matrix.hpp"
#include <iostream>
#include <vector>
template<typename T>
void printVector(const std::vector<T> &V) {
    using namespace std;
    cout << "[ ";
    for_each(V.begin(), V.end(), [](int a) {
        cout << a << " ";
    });
    cout << "]" << endl;
}
template<typename T>
class sparseMatrix {
private:
    int rows, cols;
    std::vector<int> row_index;
    std::vector<int> col_index;
    std::vector<T> elements;
    int size;
public:
    sparseMatrix(int row, int col) {
        if (row == 0 || col == 0) {
            throw std::invalid_argument("row or column cannot be zero!");
        }
        rows = row;
        cols = col;
        size = row * col;

    }
    sparseMatrix(Matrix<T> & matrix) {
        int m = matrix.getRowSize();
        int n = matrix.getColumnSize();
        if (m == 0 || n == 0) {
            throw std::invalid_argument("row or column cannot be zero!");
        }

        rows = m;
        cols = n;
        int numberOfNonZeroes = 0;
        for (int i = 0; i < m; i ++) {
            for( int j = 0; j < n; j++) {
                if (matrix.get(i,j)!= 0)
                {
                    elements.push_back(matrix.get(i, j));
                    col_index.push_back(j);
                    numberOfNonZeroes++;
                }
            }
            row_index.push_back(numberOfNonZeroes);
        }
    }
    sparseMatrix(const sparseMatrix<T> & other) {
        if (other.rows == 0 || other.cols == 0) {
            throw std::invalid_argument("row or column cannot be zero!");
        }
        rows = other.rows;
        cols = other.cols;
        row_index(other.row_index);
        col_index(other.col_index);

    }
    ~sparseMatrix() = default;
    Matrix<T> convertToDense(sparseMatrix<T> &sparseMatrix) {
        Matrix<T> mat(sparseMatrix.rows,sparseMatrix.cols);
        T arr[sparseMatrix.rows * sparseMatrix.cols];
//        std::cout << sparseMatrix.rows << std::endl;
//        std::cout << sparseMatrix.cols << std::endl;
        //        T * ptr = &arr;
        int k = 0;
        for( int i = 0; i < sparseMatrix.rows;i++) {
            for( int j = 0; j < sparseMatrix.cols; j++) {
                arr[i*sparseMatrix.cols + j ] = 0;
            }
        }
        for( int i = 0; i < sparseMatrix.rows; i++) {
            int row_elements;
            if (i == 0) {
                row_elements = sparseMatrix.row_index[0];
            } else {
                row_elements = sparseMatrix.row_index[i] - sparseMatrix.row_index[i-1];
            }

            for( int j = 0; j < row_elements; j++) {
                arr[i*sparseMatrix.cols + col_index[k]] = elements[k];
                k++;
            }
        }
//        for(int i = 0; i < sparseMatrix.rows * sparseMatrix.cols; i ++) {
//            std::cout << arr[i] << ' ';
//        }
        mat.set(sparseMatrix.rows * sparseMatrix.cols, arr);
        return mat;
    }
//    T get(int row, int col);
//    sparseMatrix & set(T val, int row, int col);
    void print() {
        printVector(elements);
        printVector(col_index);
        printVector(row_index);
    }
};







#endif //CS205PROJECT_SPARSEMATRIX_HPP

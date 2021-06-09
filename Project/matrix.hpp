#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include<string>
#include <opencv2/opencv.hpp>

//#include <omp.h>
#include<string>
#include <omp.h>
#include <string>
#include <random>

#include "complex.h"

template<typename T> 
class Vector
{
public:

    Vector(int dim)
    {
        this->dim = dim;
        vector_data = std::vector<T>(dim, static_cast<T>(0.0));
    }

    Vector(std::vector<T> vector_data)
    {
        this->dim = vector_data.size();
        this->vector_data = vector_data;
    }

    int getDim()
    {
        return dim;
    }

    std::vector<T> getData()
    {
        return vector_data;
    }

    T get(int i) const
    {
        return vector_data.at(i);
    }

    void set(int i, T value)
    {
        vector_data[i] = value;
    }
    //calculate length of vector
    static T Norm(const Vector<T> &a_vec)
    {
        T norm = static_cast<T>(0.0);
        for (int i = 0; i < a_vec.getDim(); i++)
        {
            norm += (a_vec.get(i) * a_vec.get(i));
        }
        return sqrt(norm);
    }

    static Vector<T> Normalize(const Vector<T> &a_vec)
    {
        T norm = Norm(a_vec);

        Vector<T> normalized(a_vec.getData());

        return normalized / norm;

    }

    Vector<T> operator= (const Vector<T>& a_vec)
    {
        if (this != a_vec)
        {
            dim = a_vec.getDim();
            vector_data = a_vec.getData();
        }

        return *this;
    }

    friend Vector<T> operator/ (const Vector<T> &a_vec, const T& scalar)
    {
        if (scalar == static_cast<T>(0.0))
            throw std::runtime_error("Math error: Attempted to divide by zero");

        Vector<T> quotient_vec(a_vec.getDim());
        
        for (int i = 0; i < a_vec.getDim(); i++)
            quotient_vec.set(i, a_vec.get(i) / scalar);
       
        return quotient_vec;
    }

    friend Vector<T> operator* (const T &scalar, const Vector<T> &a_vec)
    {
        Vector<T> product_vec(a_vec.getDim());

        for (int i = 0; i < a_vec.getDim(); i++)
            product_vec.set(i, scalar * a_vec.get(i));

        return product_vec;
    }

    friend Vector<T> operator* (const Vector<T> &a_vec, const T &scalar)
    {
        Vector<T> product_vec(a_vec.getDim());

        for (int i = 0; i < a_vec.getDim(); i++)
            product_vec.set(i, a_vec.get(i) * scalar);

        return product_vec;
    }

    friend Vector<T> operator+ (const Vector<T> &a_vec, const Vector<T> &b_vec)
    {
        if (a_vec.getDim() != b_vec.getDim())
            throw std::invalid_argument("Vectors are compatible");

        Vector<T> sum_vec(a_vec.getDim());

        for (int i = 0; i < a_vec.getDim(); i++)
            sum_vec.set(i, a_vec.get(i) + b_vec.get(i));

        return sum_vec;
    }

    friend Vector<T> operator- (const Vector<T> &a_vec, const Vector<T> &b_vec)
    {
        if (a_vec.getDim() != b_vec.getDim())
            throw std::invalid_argument("Vectors are compatible");

        Vector<T> difference_vec(a_vec.getDim());

        for (int i = 0; i < a_vec.getDim(); i++)
            difference_vec.set(i, a_vec.get(i) - b_vec.get(i));

        return difference_vec;
    }

    static T Dot_Product(const Vector<T> &a_vec, const Vector<T> &b_vec)
    {
        // Check that the number of dimensions match.
        if (a_vec.getDim() != b_vec.getDim())
            throw std::invalid_argument("Vector dimensions do not match.");

        // Compute the dot product.
        T dot_product = static_cast<T>(0.0);

        for (int i = 0; i < a_vec.getDim(); i++)
            dot_product += (a_vec.get(i) * b_vec.get(i));

        return dot_product;
    }

//    static Vector<T> cross(const Vector<T> &a_vec, const Vector<T> &b_vec)
//    {
//        // Check that the number of dimensions match.
//        if (a_vec.getDim() != b_vec.getDim())
//            throw std::invalid_argument("Vector dimensions do not match.");
//
//        // Check that the number of dimensions is 3.
//        /* Although the cross-product is also defined for 7 dimensions, we are
//            not going to consider that case at this time. */
//        if (a_vec.getDim() != 3)
//            throw std::invalid_argument("Vectors are not three-dimensional");
//
//        // Compute the cross product.
//        std::vector<T> cross_product;
//        resultData.push_back((a_vec.get(1) * b_vec.get(2)) - (a_vec.get(2) * b_vec.get(1)));
//        resultData.push_back(-((a_vec.get(0) * b_vec.get(2)) - (a_vec.get(2) * b_vec.get(0))));
//        resultData.push_back((a_vec.get(0) * b_vec.get(1)) - (a_vec.get(1) * b_vec.get(0)));
//
//        Vector<T> cross_product_vec(cross_product);
//        return cross_prouct_vec;
//    }
    static Vector<T> Cross_Product(const Vector<T>& a_vec, const Vector<T>& b_vec)
    {
        // Check that the number of dimensions match.
        if (a_vec.getDim() != b_vec.getDim())
            throw std::invalid_argument("Vector dimensions do not match.");

        // Check that the number of dimensions is 3.
        /* Although the cross-product is also defined for 7 dimensions, we are
            not going to consider that case at this time. */
        if (a_vec.getDim() != 3)
            throw std::invalid_argument("Vectors are not three-dimensional");

        // Compute the cross product.
        std::vector<T> cross_product;
        cross_product.push_back((a_vec.get(1) * b_vec.get(2)) - (a_vec.get(2) * b_vec.get(1)));
        cross_product.push_back(-((a_vec.get(0) * b_vec.get(2)) - (a_vec.get(2) * b_vec.get(0))));
        cross_product.push_back((a_vec.get(0) * b_vec.get(1)) - (a_vec.get(1) * b_vec.get(0)));

        Vector<T> cross_product_vec(cross_product);
        return cross_product_vec;
    }

private:

    std::vector<T> vector_data;

    int dim;

    
};

template<typename T>
class Matrix {
private:
    int row;
    int col;
    int size;
    T *mat_ptr;


    Matrix<T> ReturnSlice(std::vector<int> row, std::vector<int> col) {
        Matrix<T> SliceRes(row.size(), col.size());

        for (int i = 0; i < row.size(); ++i) {
            int r = row[i];
            for (int j = 0; j < col.size(); ++j) {
                int c = col[j];
                SliceRes.get(i, j) = get(r, c);
            }
        }

        return SliceRes;

    }

    void SliceIndex(std::vector<int> &slice, int start, int end, int step) {
        if (step < 0) {
            int tmp = start;
            start = end;
            end = tmp;
        }
        int i = start;

        while ((step > 0 && i >= start && i <= end) || (step < 0 && i >= end && i <= start)) {
            slice.push_back(i);
            i += step;
        }
    }

    void ValidRowIndex(int row) {
        if (row < 0 || row > this->row - 1) {
            throw std::range_error(std::to_string(row) + " not in range of 0 and " + std::to_string(this->row - 1));
        }
    }

    void ValidColumnIndex(int col) {
        if (col < 0 || col > this->col - 1) {
            throw std::range_error(std::to_string(col) + " not in range of 0 and " + std::to_string(this->col - 1));
        }
    }

    void CheckAxis(int axis) {
        if (axis != 1 && axis != 0) {
            throw std::invalid_argument("axis must be 1 (horizontal) or 0 (vertical)");
        }
    }

    void ValidIndex(int row, int col) {
        ValidRowIndex(row);
        ValidColumnIndex(col);
    }

public:
    Matrix(int row, int col) {
        this->row = row;
        this->col = col;
        size = row * col;
        mat_ptr = new T[row * col];
    }

    Matrix(const Matrix<T> &X)
    {
        row = X.getRow();
        col = X.getCol();
        size = X.getSize();
        mat_ptr = new T[size];

        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                get(i, j) = X.get(i, j);
    }


    ~Matrix() {

    }


    void print() {
        int index = 0;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                std::cout << std::right << std::setw(7) << mat_ptr[index++];
            }
            std::cout << std::endl;
        }
    }

    int getRowSize() {
        return row;
    }

    int getColumnSize() {
        return col;
    }

    T *getPtr() {
        return mat_ptr;
    }


    //return pointer to an entry
    T &get(int row, int col) {
        ValidIndex(row, col);
        return mat_ptr[row * this->col + col];
    }

    Matrix<T> &set(int len, T *m) {
        if (len != this->size) {
            throw std::length_error("Number of element in array must be the same with the size of matrix");
        }
        for (int i = 0; i < len; ++i) {
            mat_ptr[i] = *m;
            ++m;
        }
        return *this;
    }

    void reshape(int row, int col) {
        if (row * col != size) {
            throw std::length_error("Cannot reshape because the two matrix have different shape");
        }
        this->row = row;
        this->col = col;
    }


    Matrix<T> Slice(int rowStart, int rowEnd, int rowStep,
                    int colStart, int colEnd, int colStep) {
        if (rowStart > rowEnd) {
            throw std::invalid_argument(
                    "slice must start from smaller row to bigger row (use negative step to slice backward)");
        }

        if (colStart > colEnd) {
            throw std::invalid_argument(
                    "slice must start from smaller column to bigger row (use negative step to slice backward)");
        }
        ValidRowIndex(rowStart);
        ValidRowIndex(rowEnd);
        ValidColumnIndex(colStart);
        ValidColumnIndex(colEnd);

        std::vector<int> rowIndex;
        SliceIndex(rowIndex, rowStart, rowEnd, rowStep);
        std::vector<int> colIndex;
        SliceIndex(colIndex, colStart, colEnd, colStep);

        return ReturnSlice(rowIndex, colIndex);

    }


    T Max() {
        T maxVal = mat_ptr[0];

        for (int i = 0; i < size; ++i) {
            if (maxVal < mat_ptr[i]) {
                maxVal = mat_ptr[i];
            }
        }
        return maxVal;
    }

    Matrix<T> Max(int axis) {
        CheckAxis(axis);

        if (axis == 0) {
            Matrix<T> maxMatrix(1, col);

            int maxVal = 0;
            for (int i = 0; i < col; ++i) {
                maxVal = get(0, i);
                for (int j = 0; j < row; ++j) {
                    if (maxVal < get(j, i)) {
                        maxVal = get(j, i);
                    }
                }
                maxMatrix.get(0, i) = maxVal;
            }
            return maxMatrix;
        } else if (axis == 1) {
            Matrix<T> maxMatrix(row, 1);
            int maxVal = 0;
            for (int i = 0; i < row; ++i) {
                maxVal = get(i, 0);
                for (int j = 0; j < col; ++j) {
                    if (maxVal < get(i, j)) {
                        maxVal = get(i, j);
                    }
                }
                maxMatrix.get(i, 0) = maxVal;
            }
            return maxMatrix;
        }


    }

    T Min() {
        T maxVal = mat_ptr[0];

        for (int i = 0; i < size; ++i) {
            if (maxVal > mat_ptr[i]) {
                maxVal = mat_ptr[i];
            }
        }
        return maxVal;
    }

    Matrix<T> Min(int axis) {
        CheckAxis(axis);

        if (axis == 0) {
            Matrix<T> minMatrix(1, col);

            int maxVal = 0;
            for (int i = 0; i < col; ++i) {
                maxVal = get(0, i);
                for (int j = 0; j < row; ++j) {
                    if (maxVal > get(j, i)) {
                        maxVal = get(j, i);
                    }
                }
                minMatrix.get(0, i) = maxVal;
            }
            return minMatrix;
        } else if (axis == 1) {
            Matrix<T> minMatrix(row, 1);
            int maxVal = 0;
            for (int i = 0; i < row; ++i) {
                maxVal = get(i, 0);
                for (int j = 0; j < col; ++j) {
                    if (maxVal > get(i, j)) {
                        maxVal = get(i, j);
                    }
                }
                minMatrix.get(i, 0) = maxVal;
            }
            return minMatrix;
        }


    }

    T Avg() {
        T Avg = 0;

        for (int i = 0; i < size; ++i) {
            Avg += mat_ptr[i];
        }
        Avg /= size;
        return Avg;
    }

    Matrix<T> Avg(int axis) {
        CheckAxis(axis);

        if (axis == 0) {
            Matrix<T> AvgMatrix(1, col);

            int avg = 0;
            for (int i = 0; i < col; ++i) {
                avg = 0;
                for (int j = 0; j < row; ++j) {

                    avg += get(j, i);

                }
                avg /= row;
                AvgMatrix.get(0, i) = avg;
            }
            return AvgMatrix;
        } else if (axis == 1) {
            Matrix<T> AvgMatrix(row, 1);
            int avg = 0;
            for (int i = 0; i < row; ++i) {
                avg = 0;
                for (int j = 0; j < col; ++j) {

                    avg += get(i, j);

                }
                avg /= col;
                AvgMatrix.get(i, 0) = avg;
            }
            return AvgMatrix;
        }

    }

    T Sum() {
        T Sum = 0;

        for (int i = 0; i < size; ++i) {
            Sum += mat_ptr[i];
        }

        return Sum;
    }

    Matrix<T> Sum(int axis) {
        CheckAxis(axis);

        if (axis == 0) {
            Matrix<T> SumMatrix(1, col);

            int sum = 0;
            for (int i = 0; i < col; ++i) {
                sum = 0;
                for (int j = 0; j < row; ++j) {

                    sum += get(j, i);

                }

                SumMatrix.get(0, i) = sum;
            }
            return SumMatrix;
        } else if (axis == 1) {
            Matrix<T> SumMatrix(row, 1);
            int sum = 0;
            for (int i = 0; i < row; ++i) {
                sum = 0;
                for (int j = 0; j < col; ++j) {

                    sum += get(i, j);

                }

                SumMatrix.get(i, 0) = sum;
            }
            return SumMatrix;
        }

    }

    Matrix<T> Convolve(Matrix<T> kernel) {
        if (this->row < kernel.row || this->col < kernel.col) {
            throw std::length_error("Dimension of kernel is bigger than the left matrix");
        }
        int row = this->row - kernel.row + 1;
        int col = this->col - kernel.col + 1;

        Matrix<T> ans(row, col);

        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                T sum = T();
                for (int ii = 0; ii < kernel.row; ++ii) {
                    for (int jj = 0; jj < kernel.col; ++jj) {
                        sum += kernel.get(ii, jj) * get(i + ii, j + jj);
                    }
                }

                ans.get(i, j) = sum;

            }
        }
        return ans;

    }

    int getRow()
    {
        return row;
    }

    int getCol()
    {
        return col;
    }

    static void HaveSameDim(const Matrix<T> &X, const Matrix<T> &Y)
    {
        if (X.getCol() != Y.getCol() || X.getRow() != Y.getRow()) 
        {
            throw std::invalid_argument("The matrices don't have the same shape!");
        }
    }

    static void IsCompatible(const Matrix<T> &X, const Matrix<T> &Y)
    {
        if (X.getCol() != Y.getRow())
        {
            throw std::invalid_argument("The matrices are not compatible!");
        }
    }

//    static void IsSquare(const Matrix<T> &X)
//    {
//        if (x.row != x.col)
//            throw new std::invalid_argument("Matrice is not square!");
//    }
//
//    bool IsCloseEnough(T &x, T &y) const
//    {
//        return fabs(x - y) < 1e-9;
//    }
//
//    static void IsSymmetric(const Matrix<T> &X)
//    {
//        IsSquare(X);
//        //k is diagonal_index
//        //i is row_index
//        int k = 0;
//        while (k < col)
//        {
//            int i = k + 1;
//            while (i < row)
//            {
//                //element_prime is the entry that mirrors element across the diagonal
//                T element = get(i, k);
//                T element_prime = get(k, i);
//
//                if (!IsCloseEnough(element, element_prime))
//                    throw new std::invalid_argument("Matrix is not symmetric");
//
//                i++;
//            }
//
//            k++;
//        }
//    }
//
//    //still need to fix this one
//    static Matrix<T> IsCloseToUpperRectangular(const Matrix<T> &X)
//    {
//        /* The current matrix must have at least as many columns as rows, but note that we don't
//            actually require it to be square since we assume that the user may have combined a
//            square matrix with a vector. They would do this, for example, if they were trying to
//            solve a system of linear equations. */
//        if (m_nCols < m_nRows)
//            throw std::invalid_argument("The matrix must have at least as many columns as rows.");
//
//        /* Make a copy of the matrix data before we start. We do this because the procedure below
//            will make changes to the stored matrix data (it operates 'in place') and we don't want
//            this behaviour. Therefore we take a copy at the beginning and then we will replace the
//            modified matrix data with this copied data at the end, thus preserving the original. */
//        T *tempMatrixData;
//        tempMatrixData = new T[m_nRows * m_nCols];
//        for (int i=0; i<(m_nRows*m_nCols); ++i)
//            tempMatrixData[i] = m_matrixData[i];
//
//        // Begin the main part of the process.
//        int cRow, cCol;
//        int maxCount = 100;
//        int count = 0;
//        bool completeFlag = false;
//        while ((!completeFlag) && (count < maxCount))
//        {
//            for (int diagIndex=0; diagIndex<m_nRows; ++diagIndex)
//            {
//                // Loop over the diagonal of the matrix and ensure all diagonal elements are equal to one.
//                cRow = diagIndex;
//                cCol = diagIndex;
//
//                // Find the index of the maximum element in the current column.
//                int maxIndex = FindRowWithMaxElement(cCol, cRow);
//
//                // Now consider the column.
//                // Our aim is to set all elements BELOW the diagonal to zero.
//                for (int rowIndex=cRow+1; rowIndex<m_nRows; ++rowIndex)
//                {
//                    // If the element is already zero, move on.
//                    if (!CloseEnough(m_matrixData[Sub2Ind(rowIndex, cCol)], 0.0))
//                    {
//                        int rowOneIndex = cCol;
//
//                        // Get the value stored at the current element.
//                        T currentElementValue = m_matrixData[Sub2Ind(rowIndex, cCol)];
//
//                        // Get the value stored at (rowOneIndex, cCol)
//                        T rowOneValue = m_matrixData[Sub2Ind(rowOneIndex, cCol)];
//
//                        // If this is equal to zero, then just move on.
//                        if (!CloseEnough(rowOneValue, 0.0))
//                        {
//                            // Compute the correction factor.
//                            // (required to reduce the element at (rowIndex, cCol) to zero).
//                            T correctionFactor = -(currentElementValue / rowOneValue);
//                            MultAdd(rowIndex, rowOneIndex, correctionFactor);
//                        }
//                    }
//                }
//            }
//
//            /* Test whether we have achieved the desired result of converting the
//                matrix into row-echelon form. */
//            completeFlag = this->IsRowEchelon();
//
//            // Increment the counter.
//            count++;
//        }
//        // Form the output matrix.
//        Matrix<T> outputMatrix(m_nRows, m_nCols, m_matrixData);
//
//        // Restore the original matrix data from the copy.
//        for (int i=0; i<(m_nRows * m_nCols); ++i)
//            m_matrixData[i] = tempMatrixData[i];
//
//        // Delete the copy.
//        delete[] tempMatrixData;
//
//        return outputMatrix;
//    }
//    // need to fix bugs
//
// //additon
//    friend Matrix<T> operator+ (const Matrix<T> &X, const Matrix<T> &Y)
//    {
//        HaveSameDim(X, Y);
//        Matrix<T> Sum(X.getRow(), X.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < Sum.getRow(); i++)
//        {
//            for (int j = 0; j < Sum.getCol(); j++)
//            {
//                Sum.get(i, j) = X.get(i, j) + Y.get(i, j);
//            }
//        }
//
//        return sum;
//    }
//    //subtraction
//    friend Matrix<T> operator- (const Matrix<T> &X, const Matrix<T> &Y)
//    {
//        HaveSameDim(X, Y);
//        Matrix<T> Difference(X.getRow(), X.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < Difference.getRow(); i++)
//        {
//            for (int j = 0; j < Difference.getCol(); j++)
//            {
//                Difference.get(i, j) = x.get(i, j) - y.get(i, j);
//            }
//        }
//
//        return Difference;
//    }
//    //scalar multiplication
//    friend Matrix<T> operator* (const Matrix<T> &X, const T &scalar)
//    {
//        Matrix<T> Product(X.getRow(), X.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < Product.getRow(); i++)
//        {
//            for (int j = 0; j < Product.getCol(); j++)
//            {
//                Product.get(i, j) = X.get(i, j) * scalar;
//            }
//
//        }
//
//        return Product;
//    }
//
//    friend Matrix<T> operator* (const T &scalar, const Matrix<T> &X)
//    {
//        Matrix<T> Product(X.getRow(), x.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < Product.getRow(); i++)
//        {
//            for (int j = 0; j < Product.getCol(); j++)
//            {
//                Product.get(i, j) = scalar * X.get(i, j);
//            }
//        }
//
//        return Product;
//    }
//
//    //scalar division
//    friend Matrix<T> operator/ (const Matrix<T> &x, const T &scalar)
//    {
//        Matrix<T> Quotient(X.getRow(), X.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0 ; i < Quotient.getRow(); i++)
//        {
//            for (int j = 0; j < Quotient.getCol(); j++)
//            {
//                Quotient.get(i, j) = X.get(i, j) / scalar;
//            }
//        }
//
//        return Quotient;
//    }
//
//    //transposition
//    static Matrix<T> Transpose(const Matrix<T> &X)
//    {
//        Matrix<T> Transposed(X.getCol(), X.getRow());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < row; i++)
//        {
//            for (int j = 0; j < col; j++)
//            {
//                Transposed.get(j, i) = X.get(i, j);
//            }
//        }
//
//        reutrn Transposed;
//    }
//
//    //conjugation / tranpose conjugate?
//    Matrix<Complex> Conjugate(const Matrix<T> &X)
//    {
//        Matrix<Complex> Conjugated(X.getRow(), X.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < Conjugated.getRow(); i++)
//        {
//            for (int j = 0; j < Conjugated.getCol(); j++)
//            {
//                Complex &complex = get(i, j);
//                Conjugated.get(i, j) = new Complex(complex.getReal(), -complex.getImag());
//            }
//        }
//
//        return Conjugated;
//    }
//    //element-wise multiplication
//    static Matrix<T> Elementwise_Multiplication(const Matrix<T> &X, const Matrix<T> &Y)
//    {
//        HaveSameDim(X, Y);
//        Matrix<T> Product(X.getRow(), X.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < Product.getRow(); i++)
//            {
//                for (int j = 0; j < Product.getCol(); j++)
//                {
//                    Product.get(i, j) = X.get(i, j) * Y.get(i, j);
//                }
//            }
//        }
//
//        return Product;
//    }
//    //matiix-matrix multiplication
//    static Matrix<T> Matrix_Multiplication(const Matrix<T> &X, const Matrix<T> &Y)
//    {
//        IsCompatitble(X, Y);
//        Matrix<T> Product(X.getRow(), Y.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < X.getRow(); i++)
//        {
//            for (int j = 0; j < Y.getCol(); j++)
//            {
//                Product.get(i, j) = 0;
//                for (int k = 0; k < X.getCol(); k++)
//                {
//                    Product.get(i, j) += X.get(i, k) * Y.get(k, j);
//                }
//            }
//        }
//
//        return Product;
//    }
//
//    //matrix-vector multiplication
//    friend Vector<T> operator* (const Matrix<T> &X, const Vector<T> a_vec)
//    {
//        IsCompatible(X, a_vec);
//        Vector<T> product_vec(X.getRow());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < X.getRow(); i++)
//        {
//            product_vec.get(i) = 0;
//            for (int j = 0; j < X.getCol(); j++)
//            {
//                product_vec.get(i) += X.get(i, j) * a_vec.get(j);
//            }
//        }
//
//        return product_vec;
//    }
//
//    friend Matrix<T> operator* (const Vector<T> &a_vec, const Matrix<T> &X)
//    {
//        IsCompatible(a_vec, X);
//        Vector<T> product_vec(X.getCol());
//
//        omp_set_num_threads(4);
//        #pragma omp parallel for
//        for (int i = 0; i < X.getCol(); i++)
//        {
//            product_vec.get(i) = 0;
//            for (int j = 0; j < X.getRow(); j++)
//            {
//                product.get(i) = a_vec.get(i) * X.get(i, j);
//            }
//        }
//
//        return product_vec;
//    }
//
//    //determinant
//    static T Determinant(Matrix<T> &X)
//    {
//        IsSquare(X);
//
//        T determinant = 0;
//        int n = X.getCol();
//
//        if (n == 1)
//            return X.get(0, 0);
//        else
//        {
//            for (int j = 0; j < n ; j++)
//            {
//                determinant += (((i+j) % 2 == 0) ? 1 : -1) * X.get(0, j) * Determinant(SubMatrix(X, 0, j))
//            }
//        }
//    }
//    //find cofactor
//    static Matrix<T> SubMatrix(Matrix<T> &X, int row, int col)
//    {
//        int n = X.getCol();
//        Matrix<T> SubMatrix(n-1, n-1);
//
//        for (int i = 0; i < n; i++)
//        {
//            for (int j = 0; j < n; j++)
//            {
//                if (i != row && j != col)
//                {
//                    if (j < col && i < row)
//                        SubMatrix.get(i, j) = X.get(i, j);
//                    else if (j < col && i > row)
//                        SubMatrix.get(i-1, j) = X.get(i, j);
//                    if (j > col && i < row)
//                        SubMatrix.get(i, j-1) = X.get(i, j);
//                    else if (j > col && i > row)
//                        SubMatrix.get(i-1, j-1) = X.get(i, j);
//
//                }
//            }
//        }
//
//        return SubMatrix;
//    }
//
//    //get adjoint
//    static Matrix<T> Adjoint(Matrix<T> &X)
//    {
//        IsSquare(X);
//
//        Matrix<T> Adj(n, n);
//        int n = X.getCol();
//
//        if (n == 1 && X.get(0, 0) != static_cast<T>(0))
//        {
//            Adj.get(0, 0) = 1;
//            return Adj;
//        }
//        else if (X.get(0, 0) == static_cast<T>(0))
//        {
//            throw new std::invalid_argument("Zero matrix does not have adjoint matrix");
//        }
//
//        for (int i = 0; i < n; i++)
//        {
//            for (int j = 0; j < n; j ++)
//            {
//                Adj.get(j, i) = Determinant(SubMatrix(X, i, j));
//            }
//        }
//
//        return Adj;
//
//    }
//    //find inverse
//    static Matrix<T> Inverse(Matrix<T> &X)
//    {
//        IsSquare(x);
//
//        Matrix<T> Inv(n, n);
//        Matrix<T> Adj(n, n);
//        T det = Determinant(X);
//
//        if (det == static_cast<T>(0))
//        {
//            throw new std::invalid_argument("Matix is singular. It has no inverse!");
//        }
//
//        Adj = Adjoint(X);
//
//        Inv = Adj / det;
//
//        return Inv;
//
//    }
//
//    static void QR_Decompose(const Matrix<T> &A, Matrix<T> &Q, Matrix<T> &R)
//    {
//        Matrix<T> A_Copied = A;
//        IsSquare(A);
//
//        int n = A_Copied.getCol();
//
//        std::vector<Matrix<T>> columns;
//        for (int j = 0; j < n - 1; j++)
//        {
//            //a is column vector of A
//            //b is vector onto which we wish to reflect a
//            Vector<T> a_vec (n - j);
//            Vector<T> b_vec (n - j);
//            for (int i = j; i < n; i++)
//            {
//                a_vec.set(i-j, A_Copied.get(i, j));
//                b_vec.set(i-j, static_cast<T>(0.0));
//            }
//            b_vec.set(0, static_cast<T>(1.0));
//
//            //norm of a
//            T a_vec_norm = Norm(a);
//
//            //compute sign
//            int sign = -1;
//            if (a.get(0) < static_cast<T>(0.0))
//                sign = 1;
//
//            //compute u-vector
//            Vector<T> u = a - (sign * a_vec_norm * b);
//
//            Vector<T> n = u.Normalize();
//
//            //convert n to matrice
//            Matrix<T> n_mat (colNum - j, 1);
//            for (in i = 0; i < colNum-j; i++)
//                n_mat.set(i, 0, n.get(i));
//
//            //transpose n_mat
//            Matrix<T> trans_n_mat = n_mat.Transpose();
//
//            //create identity matrix of appropriate size
//            Matrix<T> I (colNum - j, colNum - j);
//            I.setToIdentity();
//
//            //Compute Ptemp
//            Matrix<T> Ptemp = I - static_cast<2.0> * n_mat * trans_n_mat;
//
//            //form the P matrix with original dimensions
//            Matrix<T> P (colNum, colNum);
//            P.setToIdentity();
//            for (int row = j; row < colNum; row++)
//            {
//                for (int col = j; col < colNum; col++)
//                {
//                    P.set(row, col, Ptemp.get(row - j, col - j));
//                }
//            }
//
//             //store result to columns
//            columns.push_back(P);
//
//            //Apply transformation to inputMatrix
//            A_Copied = P * A_Copied;
//        }
//
//        //compute Q
//        Matrix<T> Q_Mat = col.at(0);
//        for (int i = 1; i < colNum - 1; i++)
//        {
//            Q_Mat = Q_Mat * columns.at(i).Transpose();
//        }
//
//        Q = Q_Mat;
//
//        //compute R
//        int vectorNums = columns.size();
//        Matrix<T> R_Mat = columns.at(vectorNums - 1);
//        for (int i = vectorNums - 2; i >= 0; i--)
//        {
//            R_Mat = R_Mat * columns.at(i);
//        }
//
//        R_Mat = R_Mat * A;
//
//        R = R_Mat
//
//    }
//
//    static void Eigenvalues(const Matrix<T> &A, std::vector<T> &eigenvalues)
//    {
//        //male Copy of A
//        Matrix<T> A_Copied = A;
//
//        //verify A is square
//        A_Copied.IsSquare();
//
//        //verify A is symmetric
//        A_Copied.IsSymmetric();
//
//        int num_cols = A_Copied.getCol();
//
//        //create an identity matrix
//        Matrix<T> Identity_Matrix (num_cols, num_col);
//        Identity_Matrix.SetToIdentity();
//
//        //create matrices to store Q and R
//        Matrix<T> Q (num_cols, num_cols);
//        Matrix<T> R (num_cols, num_cols);
//
//        int max_iterations = 10e3;
//        int iteration_cnt;
//        bool continue_flag = true;
//        while ((iteration_cnt < max_iteration)  && continue_flag)
//        {
//            QR_Decompose(A_Copied, Q, R);
//
//            A_Copied = R * Q;
//
//            //check if A is close enough to being upper-triangular
//            if (A.IsCloseToUpperTrianglar())
//                continue_flag = false;
//
//            iteration_cnt++;
//        }
//
//        //eigenvalues is the diagonal elements of A
//        for (int i = 0; i < num_cols; i++)
//        {
//            eigenvalues.push_back(A.get(i, i));
//        }
//    }
//
//    //find eigenvector by inverse power iteration method
//    static void Eigenvectors(const Matrix<T> &A, const T &eigenvalue, Vector<T> &eigenvector)
//    {
//        //copy
//        Matrix<T> A_Copied = A;
//
//        //verify A is square
//        A_Copied.IsSquare();
//
//        std::random_device myRandomDevice;
//        std::mt19937 myRandomGenerator(myRandomDevice());
//        std::uniform_int_distribution<int> myDistribution(1.0, 10.0);
//
//        int n = A.getCol();
//
//        Matrix<T> Identity_Matrix(n, n);
//        Identity_Matrix.SetToIdentity();
//
//        Vector<T> v(n);
//        for(int i = 0; i < n; i++)
//            v.set(i, static_cast<T>(myDistribution(myRandomGenerator)));
//
//        int max_iteration = 100;
//        int iteration_cnt = 0;
//        T min_epsilon static_cast<T>(1e-9);
//        T epsilon = static_cast<T>(1e6);
//        Vector<T> prev_vector (n);
//        Matrix<T> Temp_Matrix (n, n);
//
//        while ((iteration_cnt < max_iteration) && (epsilon > min_epsilon))
//        {
//            prev_vector = v;
//
//            Temp_Matrix = A - (eigenvalue * Identity_Matrix);
//            Temp_Matrix.Inverse();
//            v = Temp_Matrix * v;
//            v.Normalize();
//
//            delta = (v - prev_vector).norm();
//
//            iteration_cnt++;
//        }
//
//        eigenvector = v;
//    }

    static void IsSquare(const Matrix<T> &X)
    {
        if (X.getRow() != X.getCol())
            throw new std::invalid_argument("Matrix is not square!");
    }

    static bool IsCloseEnough(T &x, T &y)
    {
        return fabs(x - y) < 1e-9;
    }

    static void IsSymmetric(const Matrix<T> &X)
    {
        IsSquare(X);
        //k is diagonal_index
        //i is row_index
        for (int k = 0; k < X.getCol(); k++)
        {
            for (int i = k + 1; i < X.getCol(); i++)
            {   
                //element_prime is the entry that mirrors element across the diagonal
                T element = X.get(i, k);
                T element_prime = X.get(k, i);
                
                if (!IsCloseEnough(element, element_prime))
                    throw new std::invalid_argument("Matrix is not symmetric");

                i++;
            }

            k++;
        }
    }

    //U stand for upper triangular
    static bool IsCloseToUEnough(const Matrix<T> &X)
    {
        IsSquare(X);

        for (int diag_index = 0; diag_index < X.getRow(); diag_index++)
        {	
            for (int i = diag_index + 1; i < X.getRow(); i++)
            {
                if (!CloseEnough(X.get(i, diag_index), 0.0))
                {
                    return false;
                }
            }							
        }
        
        return true;
    }

    Matrix <T> operator= (const Matrix<T> &X)
    {
        if (this != X)
        {
            row = X.getRow();
            col = X.getCol();
            size = X.getSize();

            if (mat_ptr)
                delete[] mat_ptr;

            mat_ptr = new T[size];
            for(int i = 0; i < row; i++)
                for (int j = 0; j < col; j++)
                    get(i, j) = X.get(i, j);
        }
    }

    //additon
    friend Matrix<T> operator+ (const Matrix<T> &X, const Matrix<T> &Y)
    {
        HaveSameDim(X, Y);
        Matrix<T> Sum(X.getRow(), X.getCol());

        for (int i = 0; i < Sum.getRow(); i++)
        {
            for (int j = 0; j < Sum.getCol(); j++)
            {
                Sum.get(i, j) = X.get(i, j) + Y.get(i, j);
            }
        }

        return Sum;
    }
    //subtraction
    friend Matrix<T> operator- (const Matrix<T> &X, const Matrix<T> &Y)
    {
        HaveSameDim(X, Y);
        Matrix<T> Difference(X.getRow(), X.getCol());

        for (int i = 0; i < Difference.getRow(); i++)
        {
            for (int j = 0; j < Difference.getCol(); j++)
            {
                Difference.get(i, j) = X.get(i, j) - Y.get(i, j);
            }
        }

        return Difference;
    }
    //scalar multiplication
    friend Matrix<T> operator* (const Matrix<T> &X, const T &scalar) 
    {
        Matrix<T> Product(X.getRow(), X.getCol());

        for (int i = 0; i < Product.getRow(); i++)
        {
            for (int j = 0; j < Product.getCol(); j++)
            {
                Product.get(i, j) = X.get(i, j) * scalar;
            }
           
        }

        return Product;
    }

    friend Matrix<T> operator* (const T &scalar, const Matrix<T> &X) 
    {
        Matrix<T> Product(X.getRow(), X.getCol());

        for (int i = 0; i < Product.getRow(); i++)
        {
            for (int j = 0; j < Product.getCol(); j++)
            {
                Product.get(i, j) = scalar * X.get(i, j);
            }
        }

        return Product;
    }

    //scalar division
    friend Matrix<T> operator/ (const Matrix<T> &X, const T &scalar) 
    {
        if (scalar == static_cast<T>(0.0))
            throw std::runtime_error("Math error: Attempted to divide by zero");

        Matrix<T> Quotient(X.getRow(), X.getCol());

        for (int i = 0 ; i < Quotient.getRow(); i++)
        {
            for (int j = 0; j < Quotient.getCol(); j++)
            {
                Quotient.get(i, j) = X.get(i, j) / scalar;
            }
        }

        return Quotient;
    }

    //transposition
    static Matrix<T> Transpose(const Matrix<T> &X) 
    {
        Matrix<T> Transposed(X.getCol(), X.getRow());

        for (int i = 0; i < X.getRow(); i++)
        {
            for (int j = 0; j < X.getCol(); j++)
            {
                Transposed.get(j, i) = X.get(i, j);
            }
        }

        return Transposed;
    }

    //conjugation / transpose conjugate?
    static Matrix<Complex> Conjugate(const Matrix<T> &X) 
    {
        Matrix<Complex> Conjugated(X.getRow(), X.getCol());

        for (int i = 0; i < Conjugated.getRow(); i++)
        {
            for (int j = 0; j < Conjugated.getCol(); j++)
            {
                Complex &complex = X.get(i, j);
                Conjugated.get(i, j) = Complex(complex.getReal(), -complex.getImag());
            } 
        }

        return Conjugated;
    }
    //element-wise multiplication
    static Matrix<T> Elementwise_Multiplication(const Matrix<T> &X, const Matrix<T> &Y) 
    {
        HaveSameDim(X, Y);
        Matrix<T> Product(X.getRow(), X.getCol());

        for (int i = 0; i < Product.getRow(); i++)
        {
            for (int j = 0; j < Product.getCol(); j++)
            {
                Product.get(i, j) = X.get(i, j) * Y.get(i, j);
            }
            
        }
        
        return Product;
    }
     
    static Matrix<T> Matrix_Multiplication(const Matrix<T> &X, const Matrix<T> &Y) 
    {
        IsCompatible(X, Y);
        Matrix<T> Product(X.getRow(), Y.getCol());

        for (int i = 0; i < X.getRow(); i++)
        {
            for (int j = 0; j < Y.getCol(); j++)
            {
                Product.get(i, j) = 0;
                for (int k = 0; k < X.getCol(); k++)
                {
                    Product.get(i, j) += X.get(i, k) * Y.get(k, j);
                }
            }
        }

        return Product;
    }

    //matrix-vector multiplication 
    friend Vector<T> operator* (const Matrix<T> &X, const Vector<T> a_vec)
    {
        if (X.getCol() != a_vec.getDim())
            throw std::invalid_argument("Matrix and vector are not compatible!");

        Vector<T> product_vec(X.getRow());

        for (int i = 0; i < X.getRow(); i++)
        {
            product_vec.set(i, 0);
            for (int j = 0; j < X.getCol(); j++)
            {
                product_vec.set(i, product_vec.get(i) + X.get(i, j) * a_vec.get(j));
            }
        }

        return product_vec;
    }
    
    friend Matrix<T> operator* (const Vector<T> &a_vec, const Matrix<T> &X)
    {
        if (a_vec.getCol() != X.getRow())
            throw std::invalid_argument("Vector and matrix are not compatible!");

        Vector<T> product_vec(X.getCol());

        for (int i = 0; i < X.getCol(); i++)
        {
            product_vec.set(i, 0);
            for (int j = 0; j < X.getRow(); j++)
            {
                product_vec.set(i, product_vec.get(i) + a_vec.get(i) * X.get(i, j));
            }
        }

        return product_vec;
    }

    //determinant
    static T Determinant(const Matrix<T> &X)
    {
        IsSquare(X);

        T determinant = 0;
        int n = X.getCol();

        if (n == 1)
            return X.get(0, 0);
        else
        {
            for (int j = 0; j < n ; j++)
            {
                determinant += (((0+j) % 2 == 0) ? 1 : -1) * X.get(0, j) * Determinant(SubMatrix(X, 0, j));
            }
        }
    }
    //find cofactor
    static Matrix<T> SubMatrix(const Matrix<T> &X, int row, int col)
    {
        int m = X.getRow();
        int n = X.getCol();
        Matrix<T> SubMatrix(m-1, n-1);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != row && j != col)
                {
                    if (j < col && i < row)
                        SubMatrix.get(i, j) = X.get(i, j);
                    else if (j < col && i > row)
                        SubMatrix.get(i-1, j) = X.get(i, j);
                    else if (j > col && i < row)
                        SubMatrix.get(i, j-1) = X.get(i, j);
                    else if (j > col && i > row)
                        SubMatrix.get(i-1, j-1) = X.get(i, j);

                }
            }
        }

        return SubMatrix;
    }

    //get adjoint
    static Matrix<T> Adjoint(const Matrix<T> &X)
    {
        IsSquare(X);

        int n = X.getCol();
        Matrix<T> Adj(n, n);

        if (n == 1 && X.get(0, 0) != static_cast<T>(0.0))
        {
            Adj.get(0, 0) = static_cast<T>(1);
            return Adj;
        } 
        else if (X.get(0, 0) == static_cast<T>(0.0))
            throw new std::invalid_argument("Zero matrix does not have adjoint matrix");

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j ++)
                Adj.get(j, i) = Determinant(SubMatrix(X, i, j));

        return Adj;

    }
    //find inverse    
    static Matrix<T> Inverse(const Matrix<T> &X)
    {
        IsSquare(X);

        int n = X.getCol();
        Matrix<T> Inv(n, n);
        Matrix<T> Adj(n, n);
        T det = Determinant(X);

        if (det == static_cast<T>(0.0))
        {
            throw new std::invalid_argument("Matrix is singular. It has no inverse!");
        }

        Adj = Adjoint(X);       

        Inv = Adj / det;

        return Inv;

    }

    static void QR_Decompose(const Matrix<T> &A, Matrix<T> &Q, Matrix<T> &R)
    {
        Matrix<T> A_Copied = A;

        IsSquare(A_Copied);

        int n = A_Copied.getCol();

        std::vector<Matrix<T>> Ps;
        for (int j = 0; j < n - 1; j++)
        {
            //a is column vector of A
            //b is vector onto which we wish to reflect a
            Vector<T> a_vec (n - j);
            Vector<T> b_vec (n - j);

            for (int i = j; i < n; i++)
            {
                a_vec.set(i-j, A_Copied.get(i, j));
                b_vec.set(i-j, static_cast<T>(0.0));
            }
            b_vec.set(0, static_cast<T>(1.0));

            //norm of a 
            T a_norm = Norm(a_vec);

            //compute sign
            int sign = -1;
            if (a_vec.get(0) < static_cast<T>(0.0))
                sign = 1;

            //compute u-vector
            Vector<T> u_vec = a_vec - (sign * a_norm * b_vec);

            //compute n-vector
            Vector<T> n_vec = Normalize(u_vec);
            
            //convert n-vector to matrix to transpose
            Matrix<T> N_Mat (n - j, 1);
            for (int i = 0; i < n - j; i++)
                N_Mat.get(i, 0) =  n_vec.get(i);
            
            //transpose n_mat
            Matrix<T> N_Mat_T = Transpose(N_Mat);

            //create identity matrix of appropriate size
            Matrix<T> I (n - j, n - j);
            I.SetToIdentity();

            //Compute P_Temp
            Matrix<T> P_Temp = I - static_cast<T>(2.0) * N_Mat * N_Mat_T;

            //form the P matrix with original dimensions
            Matrix<T> P (n, n);
            P.SetToIdentity();

            for (int row = j; row < n; row++)
                for (int col = j; col < n; col++)
                    P.get(row, col) = P_Temp.get(row - j, col - j);

            //store result to Ps
            Ps.push_back(P);

            //Apply transformation to inputMatrix
            A_Copied = P * A_Copied;
        }

        //compute Q
        Matrix<T> Q_Mat = Ps.at(0);

        for (int i = 1; i < n - 1; i++)
            Q_Mat = Q_Mat * Transpose(Ps.at(i));

        Q = Q_Mat;

        //compute R
        int p_num = Ps.size();
        Matrix<T> R_Mat = Ps.at(p_num - 1);

        for (int i = p_num - 2; i >= 0; i--)
        {
            R_Mat = R_Mat * Ps.at(i);
        }

        R_Mat = R_Mat * A;

        R = R_Mat;
        
    }

    static void Eigenvalues(const Matrix<T> &A, std::vector<T> &eigenvalues)
    {
        //male Copy of A
        Matrix<T> A_Copied = A;

        //verify A is square
        IsSquare(A_Copied);

        //verify A is symmetric
        IsSymmetric(A_Copied);

        int n = A_Copied.getCol();

        //create an identity matrix
        Matrix<T> I (n, n);
        I.SetToIdentity();

        //create matrices to store Q and R
        Matrix<T> Q (n, n);
        Matrix<T> R (n, n);

        int max_iteration = 10e3;
        int iteration_cnt = 0;
        while (iteration_cnt < max_iteration)
        {
            QR_Decompose(A_Copied, Q, R);

            A_Copied = R * Q;

            //check if A is close enough to being upper-triangular
            if (IsCloseToUEnough(A_Copied))
                break;
            
            iteration_cnt++;
        }

        //eigenvalues is the diagonal elements of A
        for (int i = 0; i < n; i++)
        {
            eigenvalues.push_back(A_Copied.get(i, i));
        }
    }

    void SetToIdentity(Matrix<T> &X)
    {
        IsSquare(X);

        for (int i = 0; i < X.getRow(); i++)
        {
            for (int j = 0; j < X.getCol(); j++)
            {
                if (i == j)
                    X.get(i, j) = 1.0;
                else
                    X.get(i, j) = 0.0;
            }
        }
    }
    //find eigenvector by inverse power iteration method
    static void Eigenvectors(const Matrix<T> &A, const T &eigenvalue, Vector<T> &eigenvector)
    {
        //copy
        Matrix<T> A_Copied = A;

        //verify A is square
        IsSquare(A_Copied);

        std::random_device myRandomDevice;
        std::mt19937 myRandomGenerator(myRandomDevice());
        std::uniform_int_distribution<int> myDistribution(1.0, 10.0);

        int n = A.getCol();

        Matrix<T> I(n, n);
        I.SetToIdentity();

        Vector<T> v_vec(n);
        for (int i = 0; i < n; i++)
            v_vec.set(i, static_cast<T>(myDistribution(myRandomGenerator)));
        
        int max_iteration = 100;
        int iteration_cnt = 0;
        T min_epsilon = static_cast<T>(1e-9);
        T epsilon = static_cast<T>(1e6);
        Vector<T> prev_vec(n);
        Matrix<T> Temp_Matrix(n, n);
        Matrix<T> Temp_Matrix_Inv(n, n);

        while ((iteration_cnt < max_iteration) && (epsilon > min_epsilon))
        {
            prev_vec = v_vec;

            Temp_Matrix = A_Copied - (eigenvalue * I);
            Temp_Matrix_Inv = Inverse(Temp_Matrix);
            v_vec = Temp_Matrix_Inv * v_vec;
            v_vec = Normalize(v_vec);

            epsilon = Norm((v_vec - prev_vec));

}

        eigenvector = v_vec;
    }
    
};
template <typename T>
cv::Mat convertToOpenCV(Matrix<T> &matrix) {
    int type;
    if (std::is_same<T, uint8_t>()) {
        type = 0;
    } else if (std::is_same<T, int8_t>()) {
        type = 1;
    } else if (std::is_same<T, uint16_t>()) {
        type = 2;
    } else if (std::is_same<T, int16_t>()) {
        type = 3;
    } else if (std::is_same<T, int32_t>()) {
        type = 4;
    } else if (std::is_same<T, float>()) {
        type = 5;
    } else if (std::is_same<T, double>()) {
        type = 6;
    } else {
        type = 7;
    }
    cv::Mat mat(matrix.getRowSize(), matrix.getColumnSize(), type, matrix.getPtr());
    return mat;
};
template <typename T>
Matrix<T> convertFromOpenCV(cv::Mat &mat) {
    T arr[mat.rows * mat.cols * mat.channels() + 1];

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols * mat.channels(); j++) {
            auto *p = mat.ptr(i, j);
            arr[i * mat.rows + j] = *p;
        }
    }
    Matrix<T> matrix(mat.rows, mat.cols * mat.channels());
    matrix.set(mat.rows * mat.cols * mat.channels(), arr);
    return matrix;
}

#endif
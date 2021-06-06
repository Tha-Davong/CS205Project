#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <omp.h>

template<typename T>
class Matrix
{
private:
    int row;
    int col;
    int size;
    T* mat_ptr;
public:
    Matrix(int row, int col) {
        this->row = row;
        this->col = col;
        size = row * col;
        mat_ptr = new T[row * col];
    }

    Matrix() {
        
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
    
    T& get(int row, int col) {
        return mat_ptr[row * this->col + col];
    }

    Matrix<T>& set(int len, T* m) {
        for (int i = 0; i < len; ++i) {
            mat_ptr[i] = *m;
            ++m;
        }
        return *this;
    }

    void reshape(int row, int col) {
        if (row * col != size) {
            //size error
        }
        this->row = row;
        this->col = col;
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
    
    Matrix<T> Slice(int rowStart, int rowEnd, int rowStep,
                    int colStart, int colEnd, int colStep) 
    {
        std::vector<int> rowIndex; 
        SliceIndex(rowIndex, rowStart, rowEnd, rowStep);
        std::vector<int> colIndex;
        SliceIndex(colIndex, colStart, colEnd, colStep);

        return ReturnSlice(rowIndex, colIndex);

    }

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
    T Max(){
        T maxVal = mat_ptr[0];

        for (int i = 0; i < size; ++i) {
            if (maxVal < mat_ptr[i]) {
                maxVal = mat_ptr[i];
            }
        }
        return maxVal;
    }

    Matrix<T> Max(int axis) {
        
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
        }
        else if (axis == 1) {
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
        }
        else if (axis == 1) {
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
        }
        else if (axis == 1) {
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
        }
        else if (axis == 1) {
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

    //get entry
    
    const T& getElement(int i) const 
    {
        return mat_ptr[i];
    }

    const T& getElement(int i, int j) const
    {
        return mat_ptr[i*row + j]
    }

    static void checkIfMatricesHaveSameDim(const Matrix<T> &x, const Matrix<T> &y) 
    {
        if (x.getCols() != y.getCols() || x.getRows() != y.getRows()) 
        {
            throw std::length_error("The matrices don't have the same shape!");
        }
    }

    static void checkIfCompatible(const Matrix<T> %x, const Matrix<T> &y)
    {
        if (x.col != y.row)
        {
            throw std::length_error("The matrices are not compatible!");
        }
    }



    //additon
    friend Matrix<T> operator+(const Matrix<T> &x, const Matrix<T> &y)
    {
        checkIfMatricesHaveSameDim(x, y);
        Matrix<T> sum(x.row, x.col);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < x.size; i++)
        {
            sum.getElement(i) = x.getElement(i) + getElement[i];
        }

        return sum;
    }
    //subtraction
    friend Matrix<T> operator-(const Matrix<T> &x, const Matrix<T> &y)
    {
        checkIfMatricesHaveSameDim(x, y);
        Matrix<T> difference(x.row, x.col);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < x.size; i++)
        
        {
            difference.getElement(i) = x.getElement(i) - y.getElement(i);
        }

        return difference;
    }
    //scalar multiplication
    friend Matrix<T> operator*(const Matrix<T> &x, const T &scalar) 
    {
        Matrix<T> product(x.row, x.col);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < x.size; i++)
        {
            product.getElement(i) = x.getElement(i) * scalar;
        }

        return product;
    }

    friend Matrix<T> operator*(const T &scalar, const Matrix<T> &x) 
    {
        Matrix<T> product(x.row, x.col);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < x.size; i++)
        {
            product.getElement(i) = scalar * x.getElement(i);
        }

        return product;
    }
    //scalar division
    friend Matrix<T> operator/(const Matrix<T> &x, const T &scalar) 
    {
        Matrix<T> quotient(x.row, x.col);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0 ; i < x.size; i++)
        {
            quotient.getElement(i) = x.getElement(i) / scalar;
        }
    }
    //transposition
    Matrix<T> transpose() const
    {
        Matrix<T> transposed(col, row);
        
        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                transposed.getElement(j, i) = getElement(i, j)
            }
        }

        reutrn transposed;
    }

    //conjugation / tranpose conjugate?
    Matrix<Complex> conjugate() const
    {
        Matrix<Complex> conjugated(row, col);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            Complex &complex = getElement(i);
            conjugated.getElement(i) = new Complex(complex.getReal(), -complex.getImag()));   
        }

        return conjugated;
    }
    //element-wise multiplication
    static Matrix<T> ele_mul(const Matrix<T> &x, const Matrix<T> &y)
    {
        checkIfMatricesHaveSameDim(x, y);
        Matrix<T> product(x.row, x.col);

        omp_set_num_threads(4);
        #pragma omp parallel for
        for (int i = 0; i < Math.min(x.size, y.size); i++)
            {
                product.getElement(i) = x.getElement(i) * y.getElement(i);
            }
        }
    }
    //matiix-matrix multiplication
    static Matrix<T> mat_mul(const Matrix<T> &x, const Matrix<T> &y)
    {
        checkIfCompatible(x, y);
        Matrix<T> product(x.row, y.col);

        omp_set_num_threads(4);
        #pragma omp parallel for private(i, j, k) shared (x, y)
        for (int i = 0; i < x.row; i++)
        {
            for (int j = 0; j < y.col; j++)
            {
                product.get(i, j) = 0;
                for (int k = 0; k < x.col; k++)
                {
                    product.get(i, j) += x.get(i, k) * y.get(k, j);
                }
            }
        }

        return product;
    }
    //matrix-vector multiplication should be the same as matrix-matrix multiplication, why need seperate
    friend Matrix<T> vec_mul(const M)
    {

    }
    //dot product
    friend T dot_product(const Vector<T> &x, const Vector<T> &y)
    {
        
    }
    //computing eigenvalues
    //determinant
    static void checkIfSquare(const Matrix<T> x) const
    {
        if (x.row != x.col)
            throw new std::length_error("Matrice is not square!");
    }

    static T determinant(Maatrix<T> &x, int n)
    {
        checkIfSquare(x);
        T determinant = 0;

        if (row == 1)
            return x.getElement(0);
        else
        {
            for (int j = 0; j < n ; j++)
            {
                determinant += (((i+j) % 2 == 0) ? 1 : -1) * x.getElement(0, j) * determinant(cofactor(x, 0, j, n), n-1)
            }
        }
    }

    static Matrix<T> cofactor(Matrix<T> &x, int row, int col, int n)
    {
        Matrix<T> cofact(n-1, n-1);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != row && j != col)
                {
                    if (j < col && i < row)
                        cofact.getElement(i, j) = x.getElement(i, j);
                    else if (j < col && i > row)
                            cofact.getElement(i-1, j) = x.getElement(i, j);
                    if (j > col && i < row)
                        cofact.getElement(i, j-1) = x.getElement(i, j);
                    else if (j > col && i > row)
                        confact.getElement(i-1, j-1) = x.getElement(i, j);

                }
            }
        }
        return cofact;
    }
    //get adjoint
    static Matrix<T> adjoint(Matrix<T> &x, int n)
    {
        checkIfSquare(x);
        Matrix<T> adj(n, n);

        if (n == 1 && x.getElement(0) != 0)
        {
            adj.get(0) = 1;
            return adj;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j ++)
            {
                adj.getElement(j, i) = determinant(confactor(x, i, j), n-1);
            }
        }

        return adj;

    }
    
    static Matrix<T> inverse(Matrix<T> &x, int n)
    {
        checkIfSquare(x);
        Matrix<T> inv(n, n);
        T det = determinant(x, n);
        Matrix<T> adj(n, n);

        if (det == 0)
        {
            throw new std::length_error("Matix is singular. It has no inverse!");
        }

        adj = adjoint(x, n);       

        inv = adj / det;

        return inv;

    }
    
};









#endif
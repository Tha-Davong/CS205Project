#ifndef CS205_PROJECT_MATRIX_HPP
#define CS205_PROJECT_MATRIX_HPP

#include <memory>
#include <complex>
#include <tuple>
#include <vector>

#include "Complex.hpp"

namespace matrix {

    void checkBound(int it, int lower, int upper);

    std::vector<int> makeSlice(int start, int end, int step, int upperBound);

    template<typename T>
    class Vector {
    private:
        int row_;
        int cols_;
        std::weak_ptr<T[]> mat_ptr_;
    public:
        Vector(int row, int cols, std::weak_ptr<T[]> mat_ptr) : row_(row), cols_(cols), mat_ptr_(mat_ptr) {}

        T &operator[](int col) {
            checkBound(col, 0, cols_);
            if (mat_ptr_.expired()) {
                throw std::bad_weak_ptr();
            }
            return mat_ptr_.lock()[row_ * cols_ + col];
        }
    };

    template<typename T>
    class Matrix {
    private:
        int rows_;
        int cols_;
        int size_;
        std::shared_ptr<T[]> mat_ptr_;
    public:
        // Initial matrix with specific rows and columns.
        Matrix(int rows, int cols) : rows_(rows), cols_(cols), size_(rows*cols), mat_ptr_(new T[rows * cols]()) {}
        Matrix(int rows, int cols, bool init) : rows_(rows), cols_(cols), size_(rows*cols), mat_ptr_(init ? new T[rows * cols]() : new T[rows * cols]) {}

        Matrix(const Matrix<T> &mat) : Matrix(mat.rows_, mat.cols_) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    unsafe(i, j) = mat.unsafe(i, j);
                }
            }
        }

        Matrix(Matrix<T> &&mat) noexcept: rows_(mat.rows_), cols_(mat.cols_), mat_ptr_(mat.mat_ptr_) {}

        

        ~Matrix() {
            mat_ptr_.reset();
        }

        [[nodiscard]] int getCols() const {
            return cols_;
        }

        [[nodiscard]] int getRows() const {
            return rows_;
        }

        void reshape(int rows, int cols) {
            if (rows * cols != size_) {
                throw std::length_error("The matrices don't have the same size!");
            }
            rows_ = rows;
            cols_ = cols;
        }

        Matrix<T> &set(int row, int col, T val) {
            checkBound(row, 0, rows_);
            checkBound(col, 0, cols_);
            mat_ptr_[row * cols_ + col] = val;
            return *this;
        }

        Matrix<T> &set(int length, T *vals) {
            for (int i = 0; i < length; ++i) {
                mat_ptr_[i] = *vals;
                ++vals;
            }
            return *this;
        }

        T &get(int row, int col) {
            checkBound(row, 0, rows_);
            checkBound(col, 0, cols_);
            return mat_ptr_[row * cols_ + col];
        }

        const T &get(int row, int col) const {
            checkBound(row, 0, rows_);
            checkBound(col, 0, cols_);
            return mat_ptr_[row * cols_ + col];
        }

        inline T &unsafe(int row, int col) {
            return mat_ptr_[row * cols_ + col];
        }

        inline const T &unsafe(int row, int col) const {
            return mat_ptr_[row * cols_ + col];
        }

        inline T &unsafe(int idx) {
            return mat_ptr_[idx];
        }

        inline const T &unsafe(int idx) const {
            return mat_ptr_[idx];
        }

        Vector<T> operator[](int row) {
            checkBound(row, 0, rows_);
            return Vector<T>(row, cols_, mat_ptr_);
        }

        Matrix<T> &operator=(const Matrix<T> &other) {
            if (this == other) {
                return *this;
            }
            rows_ = other.rows_;
            cols_ = other.cols_;
            size_ = other.size_;
            mat_ptr_.reset(new T[size_]);

            #pragma omp parallel for
            for (int i = 0; i < size_; ++i) {
                unsafe(i) = other.unsafe(i);
            }
            return *this;
        }

        Matrix<T> &operator=(Matrix<T> &&other) noexcept {
            rows_ = other.rows_;
            cols_ = other.cols_;
            size_ = other.size_;
            mat_ptr_ = other.mat_ptr_;
            return *this;
        }

        bool operator==(const Matrix<T> &other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                return false;
            }

            bool isEqual = true;

            #pragma omp parallel for reduction(&: isEqual)
            for (int i = 0; i < size_; ++i) {
                isEqual &= (unsafe(i) == other.unsafe(i));
            }
            return isEqual;
        }

        bool operator!=(const Matrix &rhs) const {
            return rhs != *this;
        }

        

        static void assertMatricesWithSameShape(const Matrix<T> &first, const Matrix<T> &second) {
            if (first.getCols() != second.getCols() || first.getRows() != second.getRows()) {
                throw std::length_error("The matrices don't have the same shape!");
            }
        }

        // add
        friend Matrix<T> operator+(const Matrix<T> &first, const Matrix<T> &second) {
            Matrix<T>::assertMatricesWithSameShape(first, second);

            matrix::Matrix<T> result(first.rows_, first.cols_, false);

            #pragma omp parallel for
            for (int i = 0; i < first.size_; ++i) {
                result.unsafe(i) = first.unsafe(i) + second.unsafe(i);
            }
            return result;
        }

        // minus
        friend Matrix<T> operator-(const Matrix<T> &first, const Matrix<T> &second) {
            Matrix<T>::assertMatricesWithSameShape(first, second);

            matrix::Matrix<T> result(first.rows_, first.cols_, false);

            #pragma omp parallel for
            for (int i = 0; i < first.size_; ++i) {
                result.unsafe(i) = first.unsafe(i) - second.unsafe(i);
            }
            return result;
        }

        // unary minus
        Matrix<T> operator-() const {
            matrix::Matrix<T> result(rows_, cols_, false);

            #pragma omp parallel for
            for (int i = 0; i < size_; ++i) {
                result.unsafe(i) = -unsafe(i);
            }
            return result;
        }

        // scalar multiplication
        friend Matrix<T> operator*(const T &val, const Matrix<T> &mat) {
            matrix::Matrix<T> result(mat.rows_, mat.cols_, false);

            #pragma omp parallel for
            for (int i = 0; i < mat.size_; ++i) {
                result.unsafe(i) = mat.unsafe(i) * val;
            }
            return result;
        }

        // scalar multiplication
        friend Matrix<T> operator*(const Matrix<T> &mat, const T &val) {
            matrix::Matrix<T> result(mat.rows_, mat.cols_, false);

            #pragma omp parallel for
            for (int i = 0; i < mat.size_; ++i) {
                result.unsafe(i) = mat.unsafe(i) * val;
            }
            return result;
        }

        // scalar division
        friend Matrix<T> operator/(const Matrix<T> &mat, const T &val) {
            matrix::Matrix<T> result(mat.rows_, mat.cols_, false);

            #pragma omp parallel for
            for (int i = 0; i < mat.size_; ++i) {
                result.unsafe(i) = mat.unsafe(i) / val;
            }
            return result;
        }

        Matrix<T> transposition() const {
            matrix::Matrix<T> result(cols_, rows_, false);

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    result.unsafe(j, i) = unsafe(i, j);
                }
            }
            return result;
        }

        Matrix<Complex> conjugation() const {
            matrix::Matrix<Complex> result = transposition();

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    const Complex &value = result.unsafe(j, i);
                    result.unsafe(j, i) = Complex(value.getReal(), -value.getImag());
                }
            }
            return result;
        }

        void getConfactor(Matrix<T> &t, int s, int e, int n) const {
            int i = 0, j = 0;
            for (int row = 0; row < n; ++row) {
                for (int col = 0; col < n; ++col) {
                    if (row != s && col != e) {
                        t.unsafe(i, j++) = unsafe(row, col);
                        if (j == n-1) j = 0, ++i;
                    }
                }
            }
        }

        T determinant(int n) const {
            if (getRows() != getCols()) return 0;
            T result = 0;
            if (n == 1) result = unsafe(0, 0);
            else {
                Matrix<T> t(n, n);
                T mul = 1;
                for (int i = 0; i < n; ++i) {
                    getConfactor(t, 0, i, n);
                    result += mul * unsafe(0, i) * t.determinant(n - 1);
                    mul = -mul;
                }
            }
            return result;
        }

        // element-wise multiplication.
        Matrix<T> multiply(const Matrix<T> &other) const {
            Matrix<T>::assertMatricesWithSameShape(*this, other);

            matrix::Matrix<T> result(rows_, cols_, false);

            #pragma omp parallel for
            for (int i = 0; i < size_; ++i) {
                result.unsafe(i) = unsafe(i) * other.unsafe(i);
            }
            return result;
        }

        // matrix-matrix multiplication.
        Matrix<T> matmul(const Matrix<T> &other) const {
            if (getCols() != other.getRows()) {
                throw std::length_error("Matrices with incompatible shapes cannot perform matmul!");
            }

            int rows = getRows();
            int cols = other.getCols();

            matrix::Matrix<T> result(rows, cols, false);

            matrix::Matrix<T> trans = other.transposition(); // convert row-major to column-major to avoid stride memory access

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    T sumVal = T();
                    #pragma omp parallel for reduction(+: sumVal)
                    for (int k = 0; k < cols_; ++k) {
                        sumVal += unsafe(i, k) * other.unsafe(j, k);
                    }
                    result.unsafe(i, j) = sumVal;
                }
            }
            return result;
        }

        Matrix<T> pad(int padRows, int padCols) const {
            int rows = rows_ + 2*padRows;
            int cols = cols_ + 2*padCols;
            matrix::Matrix<T> result(rows, cols, false);

            const T zero = T();

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int i_ = i - padRows;
                    int j_ = j - padCols;
                    result.unsafe(i, j) = (0<=i_ && i_<rows_ && 0<=j_ && j_<cols_) ? unsafe(i_, j_) : zero;
                }
            }
            return result;
        }

        Matrix<T> convolve(const Matrix<T> &knl) const {
            if (rows_ < knl.getRows() || cols_ < knl.getCols()) {
                throw std::length_error("Kernel is larger than this matrix!");
            }

            int rows = rows_ - knl.getRows() + 1;
            int cols = cols_ - knl.getCols() + 1;

            matrix::Matrix<T> result(rows, cols, false);
            matrix::Matrix<T> trans(knl.getRows(), knl.getCols(), false);

            #pragma omp parallel for
            for (int i = 0; i < knl.size_; ++i) {
                trans.unsafe(i) = knl.unsafe(knl.size_ - i - 1);
            }

            #pragma omp parallel for
            for (int dst_i = 0; dst_i < rows; ++dst_i) {
                for (int dst_j = 0; dst_j < cols; ++dst_j) {
                    T sumVal = T();
                    #pragma omp parallel for reduction(+: sumVal) collapse(2)
                    for (int i = 0; i < knl.rows_; ++i) {
                        for (int j = 0; j < knl.cols_; ++j) {
                            sumVal += trans.unsafe(i, j) * unsafe(dst_i + i, dst_j + j);
                        }
                    }
                    result.unsafe(dst_i, dst_j) = sumVal;
                }
            }

            return result;
        }

        T sum() const {
            T sumVal = T();

            #pragma omp parallel for reduction(+: sumVal)
            for (int i = 0; i < size_; ++i) {
                sumVal += unsafe(i);
            }
            return sumVal;
        }

        T avg() const {
            T avgVal = T();

            #pragma omp parallel for reduction(+: avgVal)
            for (int i = 0; i < size_; ++i) {
                avgVal += unsafe(i) / size_;
            }
            return avgVal;
        }

        T max() const {
            T maxVal = unsafe(0);

            #pragma omp parallel for reduction(max: maxVal)
            for (int i = 0; i < size_; ++i) {
                maxVal = maxVal>unsafe(i) ? maxVal : unsafe(i);
            }
            return maxVal;
        }

        T min() const {
            T minVal = unsafe(0);

            #pragma omp parallel for reduction(min: minVal)
            for (int i = 0; i < size_; ++i) {
                minVal = minVal<unsafe(i) ? minVal : unsafe(i);
            }
            return minVal;
        }

        void adjoint(Matrix<T> &t) const {
            int n = t.getRows();
            if (n == 1) {
                t.unsafe(0, 0) = 1;
                return;
            }
            int mul = 1;
            Matrix<T> temp(n, n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    getConfactor(temp, i, j, n);
                    mul = ((i + j) & 1) ? -1 : 1;
                    t.unsafe(j, i) = mul * temp.determinant(n - 1);
                }
            }
        }

        Matrix<T> inverse() const {
            if (rows_ != cols_) return Matrix<T>(0, 0);
            Matrix<T> result(rows_, cols_);
            T det = determinant(rows_);
            if (det == 0) return Matrix<T>(0, 0);
            Matrix<T> t(rows_, cols_);
            adjoint(t);
            for (int i = 0; i < rows_; ++i)
                for (int j = 0; j < cols_; ++j)
                    result.unsafe(i, j) = t.unsafe(i, j) / det;
            return result;
        }

        T trace() {
            if (rows_ != cols_) return 0;
            T result = 0;
            for (int i = 0; i < rows_; ++i)
                result = result + unsafe(i, i);
            return result;
        }

        
        Matrix<T> sliceRow(int start = 0, int end = -1, int step = 1) {
            return sliceRow(makeSlice(start, end, step, rows_));
        }

        Matrix<T> sliceRow(const std::vector<int> &rows) {
            Matrix<T> result(rows.size(), cols_, false);

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < rows.size(); ++i) {
                for (int j = 0; j < cols_; ++j) {
                    int row = rows[i];
                    checkBound(row, 0, rows_);
                    result.unsafe(i, j) = unsafe(row, j);
                }
            }
            return result;
        }

        Matrix<T> sliceCol(int start = 0, int end = -1, int step = 1) {
            return sliceCol(makeSlice(start, end, step, cols_));
        }

        Matrix<T> sliceCol(const std::vector<int> &cols) {
            Matrix<T> result(rows_, cols.size(), false);

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < cols.size(); ++i) {
                for (int j = 0; j < rows_; ++j) {
                    int col = cols[i];
                    checkBound(col, 0, cols_);
                    result.unsafe(j, i) = unsafe(j, col);
                }
            }
            return result;
        }

        Matrix<T> slice(
                int rowStart = 0, int rowEnd = -1, int rowStep = 1,
                int colStart = 0, int colEnd = -1, int colStep = 1
        ) {
            std::vector<int> rows, cols;
            #pragma omp parallel sections
            {
                #pragma omp section
                rows = makeSlice(rowStart, rowEnd, rowStep, rows_);

                #pragma omp section
                cols = makeSlice(colStart, colEnd, colStep, cols_);
            }
            return slice(rows, cols);
        }

        Matrix<T> slice(const std::vector<int> &rows, const std::vector<int> &cols) {
            Matrix<T> result(rows.size(), cols.size(), false);

            #pragma omp parallel sections
            {
                #pragma omp section
                for (int row : rows) {
                    checkBound(row, 0, rows_);
                }

                #pragma omp section
                for (int col : cols) {
                    checkBound(col, 0, cols_);
                }
            }

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < rows.size(); ++i) {
                for (size_t j = 0; j < cols.size(); ++j) {
                    int row = rows[i];
                    int col = cols[j];
                    result.unsafe(i, j) = unsafe(row, col);
                }
            }
            return result;
        }
    };
}

#endif //CS205_PROJECT_MATRIX_HPP

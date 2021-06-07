#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <iomanip>
#include <vector>

#include "complex.h"

template<typename T>
class Matrix
{
private:
    int row;
    int col;
    int size;
    T* mat_ptr;
    std::vector<std::vector<T>> matrix;
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
    int getRowSize() {
        return row;
    }
    int getColumnSize() {
        return col;
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

    
};









#endif
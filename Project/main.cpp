#include <iostream>
#include "matrix.hpp"
#include <complex>
#include "sparseMatrix.hpp"
#include "templateUtil.h"
#include <complex>
//#include <opencv2/opencv.hpp>

using namespace std;
//void testConversionSparseMatrix();
//void testConversionOpenCV();
void ConvolutionTest();
void AvgMinMaxTest();
void testTemplateUtil();
void SliceandReshapeTest();
int main() {

    ConvolutionTest();
    AvgMinMaxTest();
    SliceandReshapeTest();

    //task_3_test
    Matrix<double> M_1(3, 3);
    Matrix<double> M_2(3, 3);
    Matrix<std::complex<double>> M_C(2,2);
    Matrix<double> M_Sym(3, 3);
    //vector initialization
    std::vector<double> vector_data_1;
    vector_data_1.push_back(1.0);
    vector_data_1.push_back(2.0);
    vector_data_1.push_back(-2.0);
    std::vector<double> vector_data_2;
    vector_data_2.push_back(3.0);
    vector_data_2.push_back(-4.0);
    vector_data_2.push_back(-1.0);
    std::vector<double> eigenvalues;
    //Matrix<std::complex> M_3(2, 2);
    double m_1[] = {1.0, 2.0, -1.0, 3.0, 4.0, -7.0, 1.0, -2.0, 3.0};
    double m_2[] = {5.0, -6.0, -4.0, 7.0, 8.0, -2.0, 4.0, 6.0, 5.0};
    std::complex<double> m_c[] = {std::complex<double>(1,1), std::complex<double>(-3,-2), std::complex<double>(0,-3),std::complex<double>(4,0)};
    double m_sym[] = {5.0, -6.0, -4.0, -6.0, 8.0, -2.0, -4.0, -2.0, 5.0};
    //std::complex m_3[] =
    double scalar_1 = 9.0;
    double scalar_2 = 2.0;
    M_1.set (9, m_1);
    /*
     *  {   1.0     2.0    -1.0
     *      3.0     4.0    -7.0
     *      1.0    -2.0     3.0     }
     */
    M_2.set (9, m_2);
    /*
     * {    5.0    -6.0     -4.0
     *      7.0     8.0     -2.0
     *      4.0     6.0      5.0     }
     */
    M_C.set(4, m_c);
    M_Sym.set (9, m_sym);
    Vector<double> v_1_vec(vector_data_1);
    /*
     * {    1.0
     *      2.0
     *      -2.0    }
     */
    Vector<double> v_2_vec(vector_data_2);
    /*
     * {    3.0
     *      -4.0
     *      -1.0    }
     */

    Vector<double> eigenvector_vec(3);
   // testConversionSparseMatrix();
   // testConversionOpenCV();
    testTemplateUtil();

    cout << "testing matrix arithmetic" << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "addition" << endl;
    (M_1 + M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "subtraction" << endl;
    (M_1 - M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "scalar multiplication" << endl;
    (M_1 * scalar_1).print();
    cout << endl;
    (scalar_1 * M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "scalar division" << endl;
    (M_1 / scalar_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "transposition" << endl;
    Matrix<double>::Transpose(M_1).print();
    cout << endl;
   
    Matrix<double>::Transpose(M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "conjugate" << endl;
    cout << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "element-wise multiplication" << endl;
    Matrix<double>::Elementwise_Multiplication(M_1, M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "matrix-matrix multiplication" << endl;
    (M_1 * M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "testing vector arithmetic" << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "addition" << endl;
    (v_1_vec + v_2_vec).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "subtraction" << endl;
    (v_1_vec - v_2_vec).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "scalar multiplication" << endl;
    (v_1_vec * scalar_1).print();
    cout << endl;
    (scalar_2 * v_2_vec).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "scalar division" << endl;
    (v_1_vec / scalar_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "dot product" << endl;
    cout << (Vector<double>::Dot_Product(v_1_vec, v_2_vec)) << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "cross product" << endl;
    (Vector<double>::Cross_Product(v_1_vec, v_2_vec)).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "matrix-vector multiplication" << endl;
    (M_1 * v_1_vec).print();
    cout << endl;
    (v_1_vec * M_2).print();
    cout << endl;
    //test task_5
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "determinant" << endl;
    cout << (Matrix<double>::Determinant(M_2)) << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "Inverse" << endl;
    (Matrix<double>::Inverse(M_2)).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "eigenvalues" << endl;
    (Matrix<double>::Eigenvalues(M_Sym, eigenvalues));
    for (int i=0; i < 3; i++)
        cout << eigenvalues.at(i) << " ";
    cout << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "eigenvectors" << endl;
    for (int i = 0; i < 3; i++)
    {
        (Matrix<double>::Eigenvectors(M_Sym,eigenvalues.at(i), eigenvector_vec));
        eigenvector_vec.print();
    }
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "addition" << endl;
    (M_1 + M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "subtraction" << endl;
    (M_1 - M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "scalar multiplication" << endl;
    (M_1 * scalar_1).print();
    cout << endl;
    (scalar_1 * M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "scalar division" << endl;
    (M_1 / scalar_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "transposition" << endl;
    Matrix<double>::Transpose(M_1).print();
    cout << endl;
    Matrix<double>::Transpose(M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "conjugate" << endl;
    (Matrix<std::complex<double>>::Conjugate(M_C)).print();
    cout << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "element-wise multiplication" << endl;
    Matrix<double>::Elementwise_Multiplication(M_1, M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "matrix-matrix multiplication" << endl;
    (M_1 * M_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "testing vector arithmetic" << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "addition" << endl;
    (v_1_vec + v_2_vec).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "subtraction" << endl;
    (v_1_vec - v_2_vec).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "scalar multiplication" << endl;
    (v_1_vec * scalar_1).print();
    cout << endl;
    (scalar_2 * v_2_vec).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "scalar division" << endl;
    (v_1_vec / scalar_2).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "dot product" << endl;
    cout << (Vector<double>::Dot_Product(v_1_vec, v_2_vec)) << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "cross product" << endl;
    (Vector<double>::Cross_Product(v_1_vec, v_2_vec)).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "matrix-vector multiplication" << endl;
    (M_1 * v_1_vec).print();
    cout << endl;
    (v_1_vec * M_2).print();
    cout << endl;
    //test task_5
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "determinant" << endl;
    cout << (Matrix<double>::Determinant(M_2)) << endl;
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "Inverse" << endl;
    (Matrix<double>::Inverse(M_2)).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "Trace" << endl;
    cout << (Matrix<double>::Trace(M_2)) << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "eigenvalues" << endl;
    (Matrix<double>::Eigenvalues(M_Sym, eigenvalues));
    for (int i=0; i < 3; i++)
        cout << eigenvalues.at(i) << " ";
    cout << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "eigenvectors" << endl;
    for (int i = 0; i < 3; i++)
    {
        (Matrix<double>::Eigenvectors(M_Sym,eigenvalues.at(i), eigenvector_vec));
        eigenvector_vec.print();
    }
    cout << "----------------------------------------------------------------------------" << endl;

    return 0;
}
/*
void testConversionSparseMatrix() {
    Matrix<int> sparse(4,3);
    int sp[] = {1,0,0,0,5,0,7,0,9,0,0,0,10,0,12,3};
    sparse.set(12,sp);
    cout << "SparseMatrix: init of Matrix" << endl;
    sparse.print();
    cout << "sparseMatrix: Representation" << endl;
    sparseMatrix<int> SP(sparse);
    SP.print();
    cout << "sparseMatrix: Conversion Back to Dense" << endl;
    Matrix<int> tmp = SP.convertToDense(SP);
    tmp.print();

}
void testConversionOpenCV() {
        Matrix<int> m1(4, 4);
        int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
        m1.set(16, a);
        cout << "openCVMatrix: init of Matrix" << endl;
        m1.print();
        cout << "openCVMatrix: convert To OpenCV" << endl;
        cv::Mat mat = convertToOpenCV(m1);
        cout << mat << endl;
        cout << "openCVMatrix: convert From OpenCV" << endl;
        Matrix<int> tmp2 = convertFromOpenCV<int>(mat);
        tmp2.print();
}
*/
void testTemplateUtil() {
        
        cout << "test some template functions" << endl;
        cout << "----------------------------------------------------------------------------" << endl;
        cout << "is_same_t<float, float > : " << is_same_t<float, float > << endl; // true
        cout <<  "is_same_t <int, double> : " << is_same_t <int, double> << endl; // false
        cout << "is_arithmetic_t<std::complex<int>> : " << is_arithmetic_t<std::complex<int>> << endl; //false
        cout << "is_arithmetic_t<char *> : " << is_arithmetic_t<char *> << endl; // false
        cout << "is_arithmetic_t<int > : " << is_arithmetic_t<int > << endl; //true
        cout << "is_complex<int> : " << is_complex<int> << endl; //false
        cout << "is_complex<std::complex<int>> : " << is_complex<std::complex<int>> << endl; //true
        cout << "----------------------------------------------------------------------------" << endl;
}

void ConvolutionTest() {
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[] = { 1,2,3,4 };
    Matrix<int> m1(4, 4), m2(2, 2);
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "Convolution Test: " << endl;
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "m1: " << endl;
    m1.set(16, a);
    m1.print();
    cout << "m2: " << endl;
    m2.set(4, b);
    m2.print();
    cout << "convolution of m1 and m2: " << endl;
    m1.Convolve(m2).print();
    cout << "----------------------------------------------------------------------------" << endl;
    complex<double> c[] = { complex<double>(-2,3),complex<double>(9,2), complex<double>(7,-5), complex<double>(-2,-5),
                            complex<double>(7,8), complex<double>(3,-9), complex<double>(7,2), complex<double>(-9,1) };
    complex<double> d[] = { complex<double>(5,7),complex<double>(-9,-5)};
    Matrix<complex<double>> m3(4, 2), m4(2, 1);
    cout << "m3: " << endl;
    m3.set(8, c);
    m3.print();
    cout << "m4: " << endl;
    m4.set(2, d);
    m4.print();
    cout << "convolution of m3 and m4: " << endl;
    m3.Convolve(m4).print();
    cout << "----------------------------------------------------------------------------" << endl;
}

void AvgMinMaxTest() {
    cout << "Average(), Max(), Min(), Sum() test" << endl;
    cout << "----------------------------------------------------------------------------" << endl;
    double a[] = { 5,-14, 4, 3, -11, 3, 80, -17, 2, 7, 13, 2, -20, 15, 9, 6 };
    Matrix<double> m1(4, 4);
    m1.set(16,a);
    cout << "m1: " << endl;
    m1.print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "Average of m1 is " << m1.Avg() << endl;
    cout << "Average of m1 (axis = 0) " << endl;
    m1.Avg(0).print();
    cout << "Average of m1 (axis = 1)"  << endl;
    m1.Avg(1).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "Min of m1 is " << m1.Min() << endl;
    cout << "Min of m1 (axis = 0) " << endl;
    m1.Min(0).print();
    cout << "Min of m1 (axis = 1)" << endl;
    m1.Min(1).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "Max of m1 is " << m1.Max() << endl;
    cout << "Max of m1 (axis = 0) " << endl;
    m1.Max(0).print();
    cout << "Max of m1 (axis = 1)" << endl;
    m1.Max(1).print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "Sum of m1 is " << m1.Sum() << endl;
    cout << "Sum of m1 (axis = 0) " << endl;
    m1.Sum(0).print();
    cout << "Sum of m1 (axis = 1)" << endl;
    m1.Sum(1).print();
    cout << "----------------------------------------------------------------------------" << endl;
    
}

void SliceandReshapeTest() {
    cout << "Slice and Reshape test" << endl;
    cout << "----------------------------------------------------------------------------" << endl;
    double a[] = { 5,-14, 4, 3, -11, 3, 80, -17, 2, 7, 13, 2, -20, 15, 9, 6 };
    Matrix<double> m1(4, 4);
    m1.set(16, a);
    cout << "m1: " << endl;
    m1.print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "Reshape (4,4) to (2, 8)" << endl;
    m1.reshape(2, 8);
    m1.print();
    cout << "----------------------------------------------------------------------------" << endl;

    cout << "Slcing Reshaped matrix row(1 to 2) step 1 and col (3 to 5) step 1" << endl;
    m1.Slice(0, 1, 1, 2, 4, 1).print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "Slcing Reshaped matrix row(1 to 2) step 1 and col (3 to 5) step -1" << endl;
    m1.Slice(0, 1, 1, 2, 4, -1).print();
    cout << "----------------------------------------------------------------------------" << endl;
    cout << "Slcing Reshaped matrix row(1 to 2) step 1 and col (1 to 8) step 2" << endl;
    m1.Slice(0, 1, 1, 0, 7, 2).print();
    cout << "----------------------------------------------------------------------------" << endl;

}

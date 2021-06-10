#include <iostream>
#include "matrix.hpp"
#include <complex>
#include "sparseMatrix.hpp"
#include "templateUtil.h"
#include <complex>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {

    //this is the testing branch
    Matrix<int> m1(4, 4), m2(2, 3);
    Matrix<complex<double>> m3(4, 3);
    Matrix<int> sparse(4,3);
    int sp[] = {1,0,0,0,5,0,7,0,9,0,0,0,10,0,12,3};
    int a[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int b[] = { 1,2,3,4,5,6};
    complex < double > c[] = { complex<double>(1,1), complex<double>(2,2), complex<double>(3,3), complex<double>(4,4),
                               complex<double>(5,5), complex<double>(6,6), complex<double>(7,7), complex<double>(8,8),
                               complex<double>(9,9), complex<double>(10,10), complex<double>(11,11), complex<double>(12,12) };
    sparse.set(12,sp);
    m1.set(16, a);
    //m1.print();
    m3.set(12, c);
    m1.print();
    m1.Min(0).print();

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

    cout << "testing matrix arithmetic" << endl;
    cout << "init" << endl;
    sparse.print();
    cout << "sparseMatrix Representation" << endl;
    sparseMatrix<int> SP(sparse);
    SP.print();
    cout << "sparseMatrix: Conversion Back to Dense" << endl;
    Matrix<int> tmp = SP.convertToDense(SP);
    tmp.print();
    cout << "init" << endl;
    m1.print();
    cout << "openCVMatrix: convert To OpenCV" << endl;
    cv::Mat mat = convertToOpenCV(m1);
    cout << mat << endl;
    cout << "openCVMatrix: convert From OpenCV" << endl;
    Matrix<int> tmp2 = convertFromOpenCV<int>(mat);
    tmp2.print();
    cout << "test some template functions" << endl;
    cout << "is_same_t<float, float > : " << is_same_t<float, float > << endl; // true
    cout <<  "is_same_t <int, double> : " << is_same_t <int, double> << endl; // false
    cout << "is_arithmetic_t<std::complex<int>> : " << is_arithmetic_t<std::complex<int>> << endl; //false
    cout << "is_arithmetic_t<char *> : " << is_arithmetic_t<char *> << endl; // false
    cout << "is_arithmetic_t<int > : " << is_arithmetic_t<int > << endl; //true
    cout << "is_complex<int> : " << is_complex<int> << endl; //false
    cout << "is_complex<std::complex<int>> : " << is_complex<std::complex<int>> << endl; //true


    cout << "addition" << endl;
    (M_1 + M_2).print();

    cout << "subtraction" << endl;
    (M_1 - M_2).print();

    cout << "scalar multiplication" << endl;
    (M_1 * scalar_1).print();
    cout << endl;
    (scalar_1 * M_2).print();
    cout << "scalar division" << endl;
    (M_1 / scalar_2).print();

    cout << "transposition" << endl;
    Matrix<double>::Transpose(M_1).print();
    cout << endl;
    Matrix<double>::Transpose(M_2).print();
    cout << "conjugate" << endl;
    cout << endl;

    cout << "element-wise multiplication" << endl;
    Matrix<double>::Elementwise_Multiplication(M_1, M_2).print();

    cout << "matrix-matrix multiplication" << endl;
    (M_1 * M_2).print();

    cout << "testing vector arithmetic" << endl;

    cout << "addition" << endl;
    (v_1_vec + v_2_vec).print();

    cout << "subtraction" << endl;
    (v_1_vec - v_2_vec).print();

    cout << "scalar multiplication" << endl;
    (v_1_vec * scalar_1).print();
    cout << endl;
    (scalar_2 * v_2_vec).print();

    cout << "scalar division" << endl;
    (v_1_vec / scalar_2).print();

    cout << "dot product" << endl;
    cout << (Vector<double>::Dot_Product(v_1_vec, v_2_vec)) << endl;

    cout << "cross product" << endl;
    (Vector<double>::Cross_Product(v_1_vec, v_2_vec)).print();

    cout << "matrix-vector multiplication" << endl;
    (M_1 * v_1_vec).print();
    cout << endl;
    (v_1_vec * M_2).print();
    cout << endl;
    //test task_5
    cout << "determinant" << endl;
    cout << (Matrix<double>::Determinant(M_2)) << endl;

    cout << "Inverse" << endl;
    (Matrix<double>::Inverse(M_2)).print();

    cout << "eigenvalues" << endl;
    (Matrix<double>::Eigenvalues(M_Sym, eigenvalues));
    for (int i=0; i < 3; i++)
        cout << eigenvalues.at(i) << " ";
    cout << endl;

    cout << "eigenvectors" << endl;
    for (int i = 0; i < 3; i++)
    {
        (Matrix<double>::Eigenvectors(M_Sym,eigenvalues.at(i), eigenvector_vec));
        eigenvector_vec.print();
    }

    cout << "addition" << endl;
    (M_1 + M_2).print();

    cout << "subtraction" << endl;
    (M_1 - M_2).print();

    cout << "scalar multiplication" << endl;
    (M_1 * scalar_1).print();
    cout << endl;
    (scalar_1 * M_2).print();
    cout << "scalar division" << endl;
    (M_1 / scalar_2).print();

    cout << "transposition" << endl;
    Matrix<double>::Transpose(M_1).print();
    cout << endl;
    Matrix<double>::Transpose(M_2).print();
    cout << "conjugate" << endl;
    (Matrix<std::complex<double>>::Conjugate(M_C)).print();
    cout << endl;

    cout << "element-wise multiplication" << endl;
    Matrix<double>::Elementwise_Multiplication(M_1, M_2).print();

    cout << "matrix-matrix multiplication" << endl;
    (M_1 * M_2).print();

    cout << "testing vector arithmetic" << endl;

    cout << "addition" << endl;
    (v_1_vec + v_2_vec).print();

    cout << "subtraction" << endl;
    (v_1_vec - v_2_vec).print();

    cout << "scalar multiplication" << endl;
    (v_1_vec * scalar_1).print();
    cout << endl;
    (scalar_2 * v_2_vec).print();

    cout << "scalar division" << endl;
    (v_1_vec / scalar_2).print();

    cout << "dot product" << endl;
    cout << (Vector<double>::Dot_Product(v_1_vec, v_2_vec)) << endl;

    cout << "cross product" << endl;
    (Vector<double>::Cross_Product(v_1_vec, v_2_vec)).print();

    cout << "matrix-vector multiplication" << endl;
    (M_1 * v_1_vec).print();
    cout << endl;
    (v_1_vec * M_2).print();
    cout << endl;
    //test task_5
    cout << "determinant" << endl;
    cout << (Matrix<double>::Determinant(M_2)) << endl;

    cout << "Inverse" << endl;
    (Matrix<double>::Inverse(M_2)).print();

    cout << "Trace" << endl;
    cout << (Matrix<double>::Trace(M_2)) << endl;

    cout << "eigenvalues" << endl;
    (Matrix<double>::Eigenvalues(M_Sym, eigenvalues));
    for (int i=0; i < 3; i++)
        cout << eigenvalues.at(i) << " ";
    cout << endl;

    cout << "eigenvectors" << endl;
    for (int i = 0; i < 3; i++)
    {
        (Matrix<double>::Eigenvectors(M_Sym,eigenvalues.at(i), eigenvector_vec));
        eigenvector_vec.print();
    }

    return 0;
}

//
// Created by Vithlaithla Long on 4/6/21.
//

#ifndef CS205PROJECT_COMPLEX_H
#define CS205PROJECT_COMPLEX_H

#include <iostream>

class Complex {
private:
    double real;
    double imag;
public:
    Complex();
    Complex(double re, double im);
    Complex operator +(const Complex & other) const;
    Complex operator +(double real) const;
    Complex operator -(const Complex & other) const;
    Complex operator -(double real) const;
    Complex operator *(const Complex & other) const;
    Complex operator *(double real) const;
    Complex operator ~() const;
    const char* operator ==(const Complex & other) const;
    const char* operator ==(double real) const;
    const char* operator !=(const Complex & other) const;
    const char* operator !=(double real) const;


    friend Complex operator +(double real, const Complex & other);
    friend Complex operator -(double real, const Complex & other);
    friend Complex operator *(double real, const Complex & other);
//        friend Complex operator ~(const Complex & other);
    friend bool operator ==(double real, const Complex & other);
    friend bool operator !=(double real, const Complex & other);

    friend std::ostream & operator << (std::ostream & os, const Complex& other);
    friend std::istream & operator >> (std::istream & is, Complex& other);

    double getReal();

    double getImag();

};



#endif //CS205PROJECT_COMPLEX_H

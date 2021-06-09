//
// Created by Vithlaithla Long on 4/6/21.
//

#include "complex.h"

Complex::Complex(): real(0), imag(0) {

}

Complex::Complex(double re, double im): real(re), imag(im) {

}

Complex Complex::operator+(const Complex &other) const {
    double result_re = real+other.real;
    double result_img = imag+other.imag;
    return {result_re,result_img};
}


Complex Complex::operator+(double real) const {
    double result_re = this->real + real;
    double result_img = this->imag;
    return {result_re,result_img};
}

Complex Complex::operator-(const Complex &other) const {
    double result_re = real -other.real;
    double result_img = imag- other.imag;
    return {result_re,result_img};
}

Complex Complex::operator-(double real) const {
    double result_re = this->real - real;
    double result_img = this->imag;
    return {result_re,result_img};
}

Complex Complex::operator*(const Complex &other) const {
    double result_re = this->real*other.real - this->imag*other.imag;
    double result_img = this->real*other.imag + this->imag*other.real;
    return {result_re,result_img};
}

Complex Complex::operator*(double real) const {
    double result_re = this->real*real;
    double result_img = this->imag*real;
    return {result_re,result_img};
}

Complex Complex::operator~() const {
    double result_re = this->real;
    double result_img = - this->imag;
    return {result_re,result_img};
}

const char* Complex::operator==(const Complex &other) const {
    return ((this->real == other.real) && (this->imag == other.imag))?"true":"false";
}

const char* Complex::operator==(double real) const {
    return ((this->real == real) && (this->imag == 0))?"true":"false";
}

const char* Complex::operator!=(const Complex &other) const {
    return !((this->real == other.real) && (this->imag == other.imag))?"true": "false";
}

const char* Complex::operator!=(double real) const {
    return !((this->real == real) && (this->imag == 0))?"true":"false";
}

Complex operator+(double real, const Complex &other) {
    double result_re = real+other.real;
    double result_img = other.imag;
    return {result_re,result_img};
}

Complex operator-(double real, const Complex &other) {
    double result_re = real- other.real;
    double result_img = other.imag;
    return {result_re,result_img};
}

Complex operator*(double real, const Complex &other) {
    double result_re = real*other.real;
    double result_img = real*other.imag;
    return {result_re,result_img};
}

bool operator==(double real, const Complex &other) {
    return ((real == other.real) && (other.imag == 0));
}

bool operator!=(double real, const Complex &other) {
    return !((real == other.real) && (other.imag == 0));
}

std::ostream &operator<<(std::ostream &os, const Complex &other) {
    if (other.imag < 0) {
        os << other.real << "" << other.imag << "i";
    } else {
        os << other.real << "+" << other.imag << "i";
    }

    return os;
}

std::istream &operator>>(std::istream &is, Complex &other) {
    std::cout << "Enter real: ";
    is >> other.real;
    std::cout << "Enter imag: ";
    is >> other.imag;
    return is;
}

double Complex::getReal()
{
    return real;
}

double Complex::getImag()
{
    return imag;
}

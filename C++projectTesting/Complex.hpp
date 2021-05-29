#ifndef CS205_COMPLEX_HPP
#define CS205_COMPLEX_HPP

#include <iostream>

class Complex {
public:
    Complex();

    Complex(double re, double im);

    Complex operator~() const;

    Complex operator+(const Complex &) const;

    Complex operator-(const Complex &) const;

    Complex operator*(const Complex &) const;

    bool operator==(const Complex &) const;

    bool operator!=(const Complex &) const;

    friend std::ostream &operator<<(std::ostream &, const Complex &);

    friend void operator>>(std::istream &, Complex &);

    friend Complex operator*(double, Complex &);

    double getImag() const { return imag; }

    double getReal() const { return real; }

private:
    double imag;
    double real;
};

#endif //CS205_COMPLEX_HPP
